"""
Retrieval-Augmented Generation (RAG) module for Oracle RAG.

This module handles:
1. In-memory document parsing and semantic chunking via Docling (lazy-loaded).
2. Persistent vector storage and indexing via ChromaDB.
3. Similarity search and rulebook passage retrieval with confidence filtering.
"""

import os
import re
import tempfile
import threading
from typing import BinaryIO, Any, Callable
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

from src.config import config
from src.schemas.responses import RuleCitation

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same",
    "she", "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves"
}


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
        self._bm25_cache: dict[str, tuple[Any, list[str]]] = {}

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

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes text into alphanumeric tokens, filtering stopwords and short terms."""
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        return [w for w in words if w not in _STOPWORDS and len(w) > 1]

    def _get_or_build_bm25(
        self, tome: str | None = None
    ) -> tuple[Any, list[str]] | None:
        """
        Retrieves or constructs a BM25Okapi index for the specified tome (or all indexed tomes).
        Returns a tuple of (BM25Okapi instance, list of document ids) or None if unavailable.
        """
        if not _HAS_BM25:
            return None

        cache_key = tome if tome is not None else "__ALL__"
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key]

        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return None

            kwargs: dict[str, Any] = {"include": ["documents"]}
            if tome is not None:
                kwargs["where"] = {"source": tome}

            data = collection.get(**kwargs)
            if not isinstance(data, dict) or not data.get("documents"):
                return None

            docs = data.get("documents", [])
            ids = data.get("ids", [])
            if not isinstance(docs, list) or not isinstance(ids, list) or len(docs) == 0:
                return None

            corpus = [self._tokenize(doc) for doc in docs if isinstance(doc, str)]
            if not corpus or not any(len(c) > 0 for c in corpus):
                return None

            bm25 = BM25Okapi(corpus)
            self._bm25_cache[cache_key] = (bm25, ids)
            return self._bm25_cache[cache_key]
        except Exception:
            return None

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[tuple[RuleCitation, float, str]],
        tome: str | None = None,
    ) -> list[tuple[RuleCitation, float]]:
        """
        Re-ranks vector candidates using Reciprocal Rank Fusion (RRF) with BM25.
        Candidates must have already passed the similarity threshold gatekeeper.
        RRF formula: RRF_score = 1/(60 + rank_vector) + 1/(60 + rank_bm25)
        """
        if not candidates:
            return []

        if not _HAS_BM25:
            return [(c, d) for c, d, _ in candidates]

        bm25_info = self._get_or_build_bm25(tome)
        q_tokens = self._tokenize(query)

        if not bm25_info or not q_tokens:
            return [(c, d) for c, d, _ in candidates]

        bm25, all_ids = bm25_info
        if bm25 is None or not hasattr(bm25, "get_scores"):
            return [(c, d) for c, d, _ in candidates]

        try:
            scores = bm25.get_scores(q_tokens)
            # Map document IDs with positive BM25 scores to their BM25 ranks
            scored_docs = [
                (all_ids[idx], scores[idx])
                for idx in range(len(scores))
                if scores[idx] > 0
            ]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(scored_docs, 1)}

            fused: list[tuple[RuleCitation, float, float]] = []
            for v_rank, (citation, dist, doc_id) in enumerate(candidates, 1):
                v_score = 1.0 / (60.0 + v_rank)
                bm25_score = (
                    (1.0 / (60.0 + bm25_ranks[doc_id]))
                    if doc_id in bm25_ranks
                    else 0.0
                )
                rrf_score = v_score + bm25_score
                fused.append((citation, dist, rrf_score))

            # Sort by hybrid RRF score descending
            fused.sort(key=lambda x: x[2], reverse=True)
            return [(c, d) for c, d, _ in fused]
        except Exception:
            return [(c, d) for c, d, _ in candidates]

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
        if (
            hasattr(chunk, "meta")
            and hasattr(chunk.meta, "doc_items")
            and chunk.meta.doc_items
        ):
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
            if (
                progress_callback
                and filename.lower().endswith(".pdf")
                and total_pages > 0
            ):
                try:
                    from docling.datamodel.base_models import InputFormat

                    pdf_pipe = self.converter._get_pipeline(InputFormat.PDF)
                    if (
                        pdf_pipe
                        and hasattr(pdf_pipe, "ocr_model")
                        and pdf_pipe.ocr_model
                    ):
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
                if (
                    hasattr(chunk, "meta")
                    and hasattr(chunk.meta, "headings")
                    and chunk.meta.headings
                ):
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
            self._bm25_cache.clear()

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
        n_results: int | None = None,
        max_distance: float | None = None,
        selected_tomes: list[str] | None = None,
    ) -> list[RuleCitation]:
        """
        Retrieves the most semantically relevant rule chunks for a given question.
        Optionally filters chunks by the specified list of selected tome/source names.
        When multiple tomes are selected, performs balanced retrieval across each tome.
        Integrates BM25 lexical re-ranking via Reciprocal Rank Fusion (RRF) on top of
        threshold-qualified vector candidates.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        # If caller explicitly provided an empty list, no tomes are selected
        if selected_tomes is not None and len(selected_tomes) == 0:
            return []

        threshold = (
            max_distance if max_distance is not None else config.similarity_threshold
        )
        top_k = n_results if n_results is not None else config.rag_top_k

        # Case 1 & 2: No specific tome filter or single tome filter
        if selected_tomes is None or len(selected_tomes) <= 1:
            where_filter = None
            tome_name = None
            if selected_tomes is not None and len(selected_tomes) == 1:
                tome_name = selected_tomes[0]
                where_filter = {"source": tome_name}

            query_res = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
                where=where_filter,
            )

            raw_candidates: list[tuple[RuleCitation, float, str]] = []
            if query_res and query_res.get("documents") and query_res["documents"][0]:
                docs = query_res["documents"][0]
                metas = (
                    query_res["metadatas"][0]
                    if query_res.get("metadatas")
                    else [{}] * len(docs)
                )
                distances = (
                    query_res["distances"][0]
                    if query_res.get("distances")
                    else [0.0] * len(docs)
                )
                ids = (
                    query_res["ids"][0]
                    if query_res.get("ids")
                    else [""] * len(docs)
                )

                for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):
                    if dist <= threshold:
                        raw_candidates.append(
                            (
                                RuleCitation(
                                    source=meta.get("source", "Unknown Tome"),
                                    section=meta.get("section", "General"),
                                    page=meta.get("page"),
                                    text=doc,
                                ),
                                float(dist),
                                doc_id,
                            )
                        )

            reranked = self._rerank_candidates(query, raw_candidates, tome=tome_name)
            return [citation for citation, _ in reranked[:top_k]]

        # Case 3: Multiple tomes selected -> Relevance-Weighted Balanced Retrieval
        unique_tomes = list(dict.fromkeys(selected_tomes))
        tome_candidates: dict[str, list[tuple[RuleCitation, float]]] = {}

        for tome in unique_tomes:
            tome_res = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
                where={"source": tome},
            )
            raw_candidates: list[tuple[RuleCitation, float, str]] = []
            if tome_res and tome_res.get("documents") and tome_res["documents"][0]:
                docs = tome_res["documents"][0]
                metas = (
                    tome_res["metadatas"][0]
                    if tome_res.get("metadatas")
                    else [{}] * len(docs)
                )
                distances = (
                    tome_res["distances"][0]
                    if tome_res.get("distances")
                    else [0.0] * len(docs)
                )
                ids = (
                    tome_res["ids"][0]
                    if tome_res.get("ids")
                    else [""] * len(docs)
                )

                for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):
                    if dist <= threshold:
                        citation = RuleCitation(
                            source=meta.get("source", "Unknown Tome"),
                            section=meta.get("section", "General"),
                            page=meta.get("page"),
                            text=doc,
                        )
                        raw_candidates.append((citation, float(dist), doc_id))

            tome_candidates[tome] = self._rerank_candidates(
                query, raw_candidates, tome=tome
            )

        # Filter only tomes that have qualifying excerpts
        active_tomes = [tome for tome in unique_tomes if tome_candidates[tome]]

        # Relevance weighting: prioritize tomes whose best excerpt is closest in semantic distance
        active_tomes.sort(key=lambda t: min(dist for _, dist in tome_candidates[t]))

        # Round-robin allocation prioritizing most relevant tomes when N > top_k
        selected_by_tome: dict[str, list[RuleCitation]] = {
            tome: [] for tome in active_tomes
        }
        total_selected = 0
        changed = True

        while total_selected < top_k and changed:
            changed = False
            for tome in active_tomes:
                idx = len(selected_by_tome[tome])
                if idx < len(tome_candidates[tome]):
                    citation, _ = tome_candidates[tome][idx]
                    selected_by_tome[tome].append(citation)
                    total_selected += 1
                    changed = True
                    if total_selected >= top_k:
                        break

        # Group by tome so excerpts from the same scripture appear together for the arbiter,
        # ordered from the most relevant rulebook to the least
        balanced_citations: list[RuleCitation] = []
        for tome in active_tomes:
            balanced_citations.extend(selected_by_tome[tome])

        return balanced_citations

    def get_indexed_tomes(self) -> list[str]:
        """Returns the list of unique book/document names currently indexed in ChromaDB."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        all_meta = collection.get(include=["metadatas"])
        if not all_meta or not all_meta.get("metadatas"):
            return []
        sources = {
            m.get("source") for m in all_meta["metadatas"] if m and m.get("source")
        }
        return sorted(list(sources))

    def delete_rulebook(self, filename: str) -> bool:
        """
        Removes all chunks associated with a given rulebook/filename from ChromaDB.

        Args:
            filename: The original filename / source name (e.g. 'Dungeon_World.pdf').

        Returns:
            bool: True if deletion succeeded, False otherwise.
        """
        try:
            collection = self._get_collection()
            collection.delete(where={"source": filename})
            self._bm25_cache.clear()
            return True
        except Exception:
            return False


_pipeline_instance: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Returns the singleton RAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
