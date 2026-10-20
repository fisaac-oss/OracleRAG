"""
Retrieval-Augmented Generation (RAG) module for Oracle RAG.

This module handles:
1. In-memory document parsing and semantic chunking via Docling (lazy-loaded).
2. Persistent vector storage and indexing via ChromaDB.
3. Similarity search and rulebook passage retrieval with confidence filtering.
"""

import os
import tempfile
import threading
from typing import BinaryIO, Any, Callable
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.config import config
from src.schemas.responses import RuleCitation


class _OCRProgressWrapper:
    """Wraps an OCR model stage to stream page-by-page progress callbacks."""

    def __init__(
        self,
        wrapped_ocr: Any,
        total_pages: int,
        progress_callback: Callable[[float, str], None],
    ):
        self._wrapped = wrapped_ocr
        self.total_pages = max(1, total_pages)
        self.progress_callback = progress_callback
        self.completed_pages = 0
        self._lock = threading.Lock()

    def __call__(self, conv_res: Any, page_batch: Any):
        for page in self._wrapped(conv_res, page_batch):
            with self._lock:
                self.completed_pages += 1
                curr = self.completed_pages

            if self.progress_callback:
                ocr_ratio = min(1.0, curr / self.total_pages)
                # Scale OCR progress from 15% to 65% of the total ingestion process
                fraction = 0.15 + (ocr_ratio * 0.50)
                pct = int(ocr_ratio * 100)
                self.progress_callback(
                    fraction,
                    f"OCR processing: Page {curr} of {self.total_pages} ({pct}%)...",
                )
            yield page

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class RAGPipeline:
    """Manages document ingestion, vector storage, and similarity retrieval."""

    def __init__(self):
        self._chroma_client = None
        self._embedding_function = None
        self._collection = None
        self._converter = None
        self._chunker = None

    def _get_collection(self):
        """Lazy-initialize Chroma persistent storage and embedding function."""
        if self._collection is None:
            os.makedirs(config.chroma_db_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=config.chroma_db_dir)
            self._embedding_function = DefaultEmbeddingFunction()
            self._collection = self._chroma_client.get_or_create_collection(
                name=config.rag_collection_name,
                embedding_function=self._embedding_function,
            )
        return self._collection

    @property
    def converter(self):
        """Lazy-load Docling DocumentConverter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    @property
    def chunker(self):
        """Lazy-load Docling HybridChunker."""
        if self._chunker is None:
            from docling.chunking import HybridChunker

            self._chunker = HybridChunker()
        return self._chunker

    def _extract_page_number(self, chunk: Any) -> int | None:
        """Helper to extract page number from Docling chunk provenance if available."""
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                if hasattr(item, "prov") and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, "page_no") and prov.page_no:
                            return int(prov.page_no)
        return None

    def ingest_document(
        self,
        file_source: str | bytes | BinaryIO,
        filename: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> int:
        """
        Parses a document in-memory using Docling and indexes the resulting
        semantic chunks into persistent ChromaDB.

        Args:
            file_source: File path (str), file bytes, or file-like object.
            filename: The original filename (e.g. 'Dungeon_World.pdf').
            progress_callback: Optional callback accepting (fraction, status_text).

        Returns:
            int: The number of rule chunks successfully indexed.
        """
        if progress_callback:
            progress_callback(0.05, f"Staging '{filename}' for analysis...")

        temp_file_path = None
        try:
            # If given bytes or a file-like stream (e.g., from Streamlit uploader)
            if isinstance(file_source, (bytes, bytearray)):
                suffix = os.path.splitext(filename)[1] or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_source)
                    temp_file_path = tmp.name
                target_path = temp_file_path
            elif hasattr(file_source, "read"):
                suffix = os.path.splitext(filename)[1] or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_source.read())
                    temp_file_path = tmp.name
                target_path = temp_file_path
            else:
                target_path = str(file_source)

            # Detect page count if PDF
            page_info = ""
            total_pages = 0
            if filename.lower().endswith(".pdf"):
                try:
                    import pypdfium2 as pdfium

                    pdf = pdfium.PdfDocument(target_path)
                    total_pages = len(pdf)
                    pdf.close()
                    if total_pages > 0:
                        page_info = f" ({total_pages} pages)"
                except Exception:
                    total_pages = 0

            # 1. Parse document structure and OCR with Docling
            pdf_pipe = None
            orig_ocr = None
            if progress_callback and filename.lower().endswith(".pdf") and total_pages > 0:
                try:
                    from docling.datamodel.base_models import InputFormat

                    pdf_pipe = self.converter._get_pipeline(InputFormat.PDF)
                    if pdf_pipe and hasattr(pdf_pipe, "ocr_model") and pdf_pipe.ocr_model:
                        orig_ocr = pdf_pipe.ocr_model
                        pdf_pipe.ocr_model = _OCRProgressWrapper(
                            orig_ocr, total_pages, progress_callback
                        )
                except Exception:
                    pdf_pipe = None
                    orig_ocr = None

            try:
                if progress_callback:
                    progress_callback(
                        0.15,
                        f"Initializing Docling OCR & structure extraction{page_info}...",
                    )

                conversion_result = self.converter.convert(target_path)
            finally:
                if pdf_pipe and orig_ocr:
                    pdf_pipe.ocr_model = orig_ocr

            if progress_callback:
                progress_callback(
                    0.65, "OCR & layout analysis complete. Slicing into rule chunks..."
                )

            # 2. Generate hierarchical semantic chunks
            chunks = list(self.chunker.chunk(conversion_result.document))
            if not chunks:
                if progress_callback:
                    progress_callback(1.0, "No text chunks discovered in document.")
                return 0

            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            ids: list[str] = []

            for idx, chunk in enumerate(chunks):
                if hasattr(self.chunker, "serialize"):
                    chunk_text = self.chunker.serialize(chunk=chunk)
                else:
                    chunk_text = getattr(chunk, "text", str(chunk))

                heading = "General Rules"
                if hasattr(chunk, "meta") and hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading = " > ".join(str(h) for h in chunk.meta.headings)

                page_num = self._extract_page_number(chunk)

                documents.append(chunk_text)
                metadatas.append(
                    {
                        "source": filename,
                        "section": heading,
                        "page": page_num or 1,
                    }
                )
                ids.append(f"{filename}_{idx}")

            # 3. Ingest into persistent ChromaDB
            if progress_callback:
                progress_callback(
                    0.85, f"Inscribing {len(documents)} rule fragments into ChromaDB..."
                )

            collection = self._get_collection()
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            if progress_callback:
                progress_callback(
                    1.0, f"Successfully inscribed {len(documents)} fragments!"
                )

            return len(documents)

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    def retrieve_relevant_rules(
        self,
        query: str,
        n_results: int = 4,
        max_distance: float | None = None,
        selected_tomes: list[str] | None = None,
    ) -> list[RuleCitation]:
        """
        Retrieves the most semantically relevant rule chunks for a given question.
        Optionally filters chunks by the specified list of selected tome/source names.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        # If caller explicitly provided an empty list, no tomes are selected
        if selected_tomes is not None and len(selected_tomes) == 0:
            return []

        threshold = max_distance if max_distance is not None else config.similarity_threshold

        where_filter = None
        if selected_tomes is not None:
            if len(selected_tomes) == 1:
                where_filter = {"source": selected_tomes[0]}
            else:
                where_filter = {"source": {"$in": selected_tomes}}

        query_res = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where_filter,
        )

        citations: list[RuleCitation] = []
        if query_res and query_res.get("documents") and query_res["documents"][0]:
            docs = query_res["documents"][0]
            metas = query_res["metadatas"][0] if query_res.get("metadatas") else [{}] * len(docs)
            distances = (
                query_res["distances"][0] if query_res.get("distances") else [0.0] * len(docs)
            )

            for doc, meta, dist in zip(docs, metas, distances):
                if dist <= threshold:
                    citations.append(
                        RuleCitation(
                            source=meta.get("source", "Unknown Tome"),
                            section=meta.get("section", "General"),
                            page=meta.get("page"),
                            text=doc,
                        )
                    )

        return citations

    def get_indexed_tomes(self) -> list[str]:
        """Returns the list of unique book/document names currently indexed in ChromaDB."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        all_meta = collection.get(include=["metadatas"])
        if not all_meta or not all_meta.get("metadatas"):
            return []
        sources = {m.get("source") for m in all_meta["metadatas"] if m and m.get("source")}
        return sorted(list(sources))


_pipeline_instance: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Returns the singleton RAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
