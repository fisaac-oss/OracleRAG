from unittest.mock import MagicMock
import pytest

from src.config import config
from src.models.model_client import (
    ModelClientError,
    OllamaConnectionError,
    ModelNotFoundError,
)
from src.services.ai_service import AIService, generate_response


def test_config_loading():
    """Verify that configuration settings load expected string values from environment or defaults."""
    assert config.ollama_base_url is not None
    assert config.model_name is not None
    assert isinstance(config.ollama_base_url, str)
    assert isinstance(config.model_name, str)


def test_empty_input_validation():
    """Verify that empty or whitespace inputs are rejected gracefully without invoking the model."""
    mock_client = MagicMock()
    service = AIService(model_client=mock_client)

    # Test empty string
    response_empty = service.process_message("")
    assert response_empty.success is False
    assert "Please enter a message" in response_empty.content
    assert mock_client.generate.call_count == 0

    # Test whitespace string
    response_spaces = service.process_message("   ")
    assert response_spaces.success is False
    assert mock_client.generate.call_count == 0


def test_successful_response_generation():
    """Verify that valid input invokes model client and returns the generated structured response."""
    mock_client = MagicMock()
    mock_client.model_name = "test_model"
    mock_client.generate.return_value = "Hello! I am an AI assistant."
    mock_rag = MagicMock()
    mock_rag.retrieve_relevant_rules.return_value = []
    mock_rag.get_indexed_tomes.return_value = []

    service = AIService(model_client=mock_client, rag=mock_rag)

    response = generate_response("Hello, AI", service=service)

    assert response.content == "Hello! I am an AI assistant."
    assert response.model == "test_model"
    assert response.elapsed_time is not None
    assert isinstance(response.elapsed_time, float)
    assert response.elapsed_time >= 0.0
    mock_client.generate.assert_called_once_with("Hello, AI")


def test_ollama_connection_error_handling():
    """Verify that Ollama connection failures produce a friendly application-level error message."""
    mock_client = MagicMock()
    mock_client.model_name = "test_model"
    mock_client.generate.side_effect = OllamaConnectionError(
        "Connection refused at http://localhost:11434"
    )
    mock_rag = MagicMock(
        get_indexed_tomes=MagicMock(return_value=[]),
        retrieve_relevant_rules=MagicMock(return_value=[]),
    )
    service = AIService(model_client=mock_client, rag=mock_rag)

    response = service.process_message("Test message")

    assert response.success is False
    assert "Could not connect to Ollama" in response.content
    assert "Connection refused" in response.error_message


def test_model_not_found_error_handling():
    """Verify that missing model exceptions are converted into controlled user messages."""
    mock_client = MagicMock()
    mock_client.model_name = "llama3.2"
    mock_client.generate.side_effect = ModelNotFoundError(
        "Model llama3.2 not found locally"
    )
    mock_rag = MagicMock(
        get_indexed_tomes=MagicMock(return_value=[]),
        retrieve_relevant_rules=MagicMock(return_value=[]),
    )
    service = AIService(model_client=mock_client, rag=mock_rag)

    response = service.process_message("Test message")

    assert response.success is False
    assert "configured AI model is unavailable" in response.content
    assert "llama3.2 not found" in response.error_message


def test_rag_grounded_prompt_and_citations():
    """Verify that retrieved rulebook passages are formatted into the prompt and returned as citations."""
    from src.schemas.responses import RuleCitation

    mock_client = MagicMock()
    mock_client.model_name = "qwen3.5:4b"
    mock_client.generate.return_value = "Grappling requires an Athletics check against Acrobatics/Athletics."

    mock_rag = MagicMock()
    citation = RuleCitation(
        source="PHB.pdf",
        section="Combat > Grappling",
        page=195,
        text="When you want to grab a creature, make an Athletics check.",
    )
    mock_rag.retrieve_relevant_rules.return_value = [citation]
    mock_rag.get_indexed_tomes.return_value = ["PHB.pdf"]

    service = AIService(model_client=mock_client, rag=mock_rag)
    response = service.process_message("How do I grapple?")

    assert response.success is True
    assert len(response.citations) == 1
    assert response.citations[0].source == "PHB.pdf"
    assert response.citations[0].page == 195
    # Verify model was called with the grounded prompt
    generated_prompt = mock_client.generate.call_args[0][0]
    assert "RULE EXCERPTS" in generated_prompt
    assert "PHB.pdf" in generated_prompt
    assert "How do I grapple?" in generated_prompt


def test_rag_fallback_refusal_guardrail():
    """Verify that when tomes are indexed but no relevant rules match, the model refuses to guess."""
    mock_client = MagicMock()
    mock_client.model_name = "qwen3.5:4b"

    mock_rag = MagicMock()
    # Tomes exist, but zero citations matched confidence threshold
    mock_rag.get_indexed_tomes.return_value = ["Dungeon_World.pdf"]
    mock_rag.retrieve_relevant_rules.return_value = []

    service = AIService(model_client=mock_client, rag=mock_rag)
    response = service.process_message("How do quantum thrusters work?")

    assert response.success is True
    assert "The consulted scriptures contain no decrees regarding this matter" in response.content
    # Crucial: LLM was never called to prevent hallucinations
    assert mock_client.generate.call_count == 0


def test_ingest_rulebook_success():
    """Verify that ingest_rulebook orchestrates RAG ingestion and returns structured IngestionResponse."""
    mock_rag = MagicMock()
    mock_rag.ingest_document.return_value = 42

    service = AIService(rag=mock_rag)
    res = service.ingest_rulebook(b"PDF content", "rules.pdf")

    assert res.success is True
    assert res.chunks_count == 42
    assert res.filename == "rules.pdf"
    mock_rag.ingest_document.assert_called_once_with(b"PDF content", "rules.pdf")


def test_model_client_temperature_options():
    """Verify that OllamaModelClient passes the configured temperature in options to ollama.chat."""
    from unittest.mock import patch
    from src.models.model_client import OllamaModelClient

    with patch("ollama.Client") as mock_ollama_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": "Test response"}}
        mock_ollama_client_cls.return_value = mock_instance

        # 1. Default temperature from config
        client = OllamaModelClient(model_name="test_model")
        client.generate("Hello world")

        mock_instance.chat.assert_called_with(
            model="test_model",
            messages=[{"role": "user", "content": "Hello world"}],
            options={"temperature": config.model_temperature},
        )

        # 2. Dynamic override temperature
        client.generate("Hello again", temperature=0.5)
        mock_instance.chat.assert_called_with(
            model="test_model",
            messages=[{"role": "user", "content": "Hello again"}],
            options={"temperature": 0.5},
        )


def test_rag_pipeline_retrieve_relevant_rules_filtering():
    """Verify that RAGPipeline applies correct ChromaDB where filter based on selected_tomes."""
    from src.capabilities.rag import RAGPipeline

    pipeline = RAGPipeline()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "documents": [["Rule excerpt about armor"]],
        "metadatas": [[{"source": "Armor_Guide.pdf", "section": "Armor", "page": 10}]],
        "distances": [[0.5]],
    }
    pipeline._collection = mock_collection

    # 1. No filter (selected_tomes=None)
    res_none = pipeline.retrieve_relevant_rules("What is heavy armor?", selected_tomes=None)
    assert len(res_none) == 1
    mock_collection.query.assert_called_with(
        query_texts=["What is heavy armor?"],
        n_results=config.rag_top_k,
        where=None,
    )

    # 2. Empty selection (selected_tomes=[]) -> should return empty without querying collection
    mock_collection.query.reset_mock()
    res_empty = pipeline.retrieve_relevant_rules("What is heavy armor?", selected_tomes=[])
    assert res_empty == []
    assert mock_collection.query.call_count == 0

    # 3. Single selected tome -> should filter where={"source": "Armor_Guide.pdf"}
    mock_collection.query.reset_mock()
    res_single = pipeline.retrieve_relevant_rules(
        "What is heavy armor?", selected_tomes=["Armor_Guide.pdf"]
    )
    assert len(res_single) == 1
    mock_collection.query.assert_called_with(
        query_texts=["What is heavy armor?"],
        n_results=config.rag_top_k,
        where={"source": "Armor_Guide.pdf"},
    )

    # 4. Multiple selected tomes -> should perform balanced retrieval per tome
    mock_collection.query.reset_mock()
    res_multi = pipeline.retrieve_relevant_rules(
        "What is heavy armor?", selected_tomes=["Armor_Guide.pdf", "PHB.pdf"]
    )
    assert len(res_multi) >= 1
    assert mock_collection.query.call_count == 2
    calls = mock_collection.query.call_args_list
    assert calls[0].kwargs["where"] == {"source": "Armor_Guide.pdf"}
    assert calls[1].kwargs["where"] == {"source": "PHB.pdf"}


def test_rag_pipeline_balanced_multi_tome_allocation():
    """Verify that RAGPipeline balances retrieval evenly across multiple tomes."""
    from src.capabilities.rag import RAGPipeline

    pipeline = RAGPipeline()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100

    def fake_query(query_texts, n_results, where):
        source = where.get("source") if where else None
        if source == "TomeA.pdf":
            return {
                "documents": [["Doc A1", "Doc A2", "Doc A3", "Doc A4"]],
                "metadatas": [[{"source": "TomeA.pdf", "section": "SecA", "page": 1}] * 4],
                "distances": [[0.3, 0.4, 0.5, 0.6]],
            }
        elif source == "TomeB.pdf":
            return {
                "documents": [["Doc B1", "Doc B2", "Doc B3", "Doc B4"]],
                "metadatas": [[{"source": "TomeB.pdf", "section": "SecB", "page": 2}] * 4],
                "distances": [[0.35, 0.45, 0.55, 0.65]],
            }
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    mock_collection.query.side_effect = fake_query
    pipeline._collection = mock_collection

    # 1. Even distribution test: top_k=6 across 2 tomes -> 3 from Tome A, 3 from Tome B
    citations = pipeline.retrieve_relevant_rules(
        "combat rules", n_results=6, selected_tomes=["TomeA.pdf", "TomeB.pdf"]
    )
    assert len(citations) == 6
    tome_a_cits = [c for c in citations if c.source == "TomeA.pdf"]
    tome_b_cits = [c for c in citations if c.source == "TomeB.pdf"]
    assert len(tome_a_cits) == 3
    assert len(tome_b_cits) == 3

    # 2. Uneven candidate availability: Tome A has 1 candidate, Tome B has 4
    def fake_query_uneven(query_texts, n_results, where):
        source = where.get("source") if where else None
        if source == "TomeA.pdf":
            return {
                "documents": [["Doc A1"]],
                "metadatas": [[{"source": "TomeA.pdf", "section": "SecA", "page": 1}]],
                "distances": [[0.3]],
            }
        elif source == "TomeB.pdf":
            return {
                "documents": [["Doc B1", "Doc B2", "Doc B3", "Doc B4"]],
                "metadatas": [[{"source": "TomeB.pdf", "section": "SecB", "page": 2}] * 4],
                "distances": [[0.35, 0.45, 0.55, 0.65]],
            }
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    mock_collection.query.side_effect = fake_query_uneven
    citations_uneven = pipeline.retrieve_relevant_rules(
        "combat rules", n_results=4, selected_tomes=["TomeA.pdf", "TomeB.pdf"]
    )
    assert len(citations_uneven) == 4
    tome_a_cits_u = [c for c in citations_uneven if c.source == "TomeA.pdf"]
    tome_b_cits_u = [c for c in citations_uneven if c.source == "TomeB.pdf"]
    assert len(tome_a_cits_u) == 1
    assert len(tome_b_cits_u) == 3

    # 3. Relevance weighting when tomes exceed top_k (N > K):
    # 4 tomes selected, top_k = 2. The 2 most relevant tomes should win the 2 slots.
    def fake_query_many(query_texts, n_results, where):
        source = where.get("source") if where else None
        distances = {
            "TomePoor1.pdf": 0.95,
            "TomeBest.pdf": 0.15,
            "TomeSecondBest.pdf": 0.40,
            "TomePoor2.pdf": 1.10,
        }
        if source in distances:
            dist = distances[source]
            return {
                "documents": [[f"Doc from {source}"]],
                "metadatas": [[{"source": source, "section": "Sec", "page": 1}]],
                "distances": [[dist]],
            }
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    mock_collection.query.side_effect = fake_query_many
    many_tomes = ["TomePoor1.pdf", "TomeBest.pdf", "TomeSecondBest.pdf", "TomePoor2.pdf"]
    citations_weighted = pipeline.retrieve_relevant_rules(
        "combat rules", n_results=2, selected_tomes=many_tomes
    )
    assert len(citations_weighted) == 2
    # TomeBest (0.15) and TomeSecondBest (0.40) must be the ones chosen!
    sources = [c.source for c in citations_weighted]
    assert sources == ["TomeBest.pdf", "TomeSecondBest.pdf"]


def test_process_message_with_selected_tomes():
    """Verify that process_message forwards selected_tomes to RAG retrieval."""
    from src.schemas.responses import RuleCitation

    mock_client = MagicMock()
    mock_client.model_name = "test_model"
    mock_client.generate.return_value = "Spellcasting rules verdict."

    mock_rag = MagicMock()
    mock_rag.get_indexed_tomes.return_value = ["Spells.pdf", "Martial.pdf"]
    citation = RuleCitation(
        source="Spells.pdf",
        section="Magic",
        page=42,
        text="A wizard can cast spells using slots.",
    )
    mock_rag.retrieve_relevant_rules.return_value = [citation]

    service = AIService(model_client=mock_client, rag=mock_rag)
    response = service.process_message(
        "How do spell slots work?",
        selected_tomes=["Spells.pdf"],
    )

    assert response.success is True
    assert len(response.citations) == 1
    assert response.citations[0].source == "Spells.pdf"
    mock_rag.retrieve_relevant_rules.assert_called_once_with(
        "How do spell slots work?",
        selected_tomes=["Spells.pdf"],
    )


def test_process_message_empty_selected_tomes_guardrail():
    """Verify that if tomes are indexed but user selects none, a friendly warning is returned without LLM call."""
    mock_client = MagicMock()
    mock_rag = MagicMock()
    mock_rag.get_indexed_tomes.return_value = ["Spells.pdf", "Martial.pdf"]
    mock_rag.retrieve_relevant_rules.return_value = []

    service = AIService(model_client=mock_client, rag=mock_rag)
    response = service.process_message(
        "How do spell slots work?",
        selected_tomes=[],
    )

    assert response.success is True
    assert "No grimoires or scriptures have been selected for consultation" in response.content
    assert mock_client.generate.call_count == 0


def test_generate_response_with_selected_tomes_and_fallback():
    """Verify generate_response helper passes selected_tomes and informs user when selected tomes have no decrees."""
    mock_client = MagicMock()
    mock_rag = MagicMock()
    mock_rag.get_indexed_tomes.return_value = ["Spells.pdf", "Martial.pdf"]
    mock_rag.retrieve_relevant_rules.return_value = []

    service = AIService(model_client=mock_client, rag=mock_rag)
    response = generate_response(
        "How does spaceship piloting work?",
        selected_tomes=["Spells.pdf"],
        service=service,
    )

    assert response.success is True
    assert "The consulted scriptures contain no decrees regarding this matter" in response.content
    assert "based on your selected tomes" in response.content
    assert mock_client.generate.call_count == 0


def test_ingest_rulebook_with_progress_callback():
    """Verify that ingest_rulebook forwards progress_callback to rag.ingest_document."""
    mock_rag = MagicMock()
    mock_rag.ingest_document.return_value = 15

    progress_calls = []

    def callback(percent: float, msg: str):
        progress_calls.append((percent, msg))

    service = AIService(rag=mock_rag)
    res = service.ingest_rulebook(
        b"PDF data", "new_rules.pdf", progress_callback=callback
    )

    assert res.success is True
    assert res.chunks_count == 15
    mock_rag.ingest_document.assert_called_once_with(
        b"PDF data", "new_rules.pdf", progress_callback=callback
    )


def test_rag_pipeline_ingest_document_invokes_progress_callback():
    """Verify that RAGPipeline.ingest_document calls the progress_callback at each major stage."""
    from src.capabilities.rag import RAGPipeline

    pipeline = RAGPipeline()
    mock_converter = MagicMock()
    mock_conv_result = MagicMock()
    mock_conv_result.document = MagicMock()
    mock_converter.convert.return_value = mock_conv_result
    pipeline._converter = mock_converter

    mock_chunker = MagicMock()
    mock_chunk = MagicMock()
    mock_chunker.chunk.return_value = [mock_chunk]
    mock_chunker.serialize.return_value = "Sample rule chunk text"
    pipeline._chunker = mock_chunker

    mock_collection = MagicMock()
    pipeline._collection = mock_collection

    progress_events = []

    def on_progress(percent: float, message: str):
        progress_events.append((percent, message))

    count = pipeline.ingest_document(
        b"dummy pdf bytes",
        "sample.pdf",
        progress_callback=on_progress,
    )

    assert count == 1
    assert len(progress_events) >= 4
    # Check that progress starts near 0 and reaches 1.0
    fractions = [p[0] for p in progress_events]
    assert fractions[0] <= 0.1
    assert fractions[-1] == 1.0
    # Check that OCR stage is reported
    messages = " ".join(p[1] for p in progress_events)
    assert "OCR" in messages
    assert "ChromaDB" in messages


def test_ocr_progress_wrapper_page_by_page():
    """Verify that _OCRProgressWrapper intercepts each page and reports granular OCR progress."""
    from src.capabilities.rag import _OCRProgressWrapper

    dummy_pages = [MagicMock(page_no=1), MagicMock(page_no=2), MagicMock(page_no=3)]

    def mock_ocr(conv_res, page_batch):
        for p in dummy_pages:
            yield p

    progress_log = []

    def callback(percent: float, message: str):
        progress_log.append((percent, message))

    wrapper = _OCRProgressWrapper(
        wrapped_ocr=mock_ocr,
        total_pages=3,
        progress_callback=callback,
    )

    yielded = list(wrapper(None, dummy_pages))
    assert len(yielded) == 3
    assert len(progress_log) == 3

    # Verify that each page was reported with increasing percentages
    assert "Page 1 of 3 (33%)" in progress_log[0][1]
    assert "Page 2 of 3 (66%)" in progress_log[1][1]
    assert "Page 3 of 3 (100%)" in progress_log[2][1]
    assert progress_log[0][0] < progress_log[1][0] < progress_log[2][0]






def test_delete_rulebook_rag_pipeline():
    """Verify that RAGPipeline.delete_rulebook deletes chunks matching the source from ChromaDB."""
    from src.capabilities.rag import RAGPipeline

    pipeline = RAGPipeline()
    mock_collection = MagicMock()
    pipeline._collection = mock_collection

    res = pipeline.delete_rulebook("test_book.pdf")

    assert res is True
    mock_collection.delete.assert_called_once_with(where={"source": "test_book.pdf"})


def test_delete_rulebook_ai_service():
    """Verify that AIService and delete_rulebook_file delegate to the RAG pipeline."""
    from src.services.ai_service import delete_rulebook_file

    mock_rag = MagicMock()
    mock_rag.delete_rulebook.return_value = True

    service = AIService(rag=mock_rag)
    res = service.delete_rulebook("custom_rules.pdf")

    assert res is True
    mock_rag.delete_rulebook.assert_called_once_with("custom_rules.pdf")

    # Test top-level function
    mock_rag.reset_mock()
    res_helper = delete_rulebook_file("custom_rules.pdf", service=service)
    assert res_helper is True
    mock_rag.delete_rulebook.assert_called_once_with("custom_rules.pdf")

    
def test_conversation_history_is_included_in_prompt():
    """Verify that recent conversation history is passed into the model prompt."""
    mock_client = MagicMock()
    mock_client.model_name = "test_model"
    mock_client.generate.return_value = "It uses an Athletics check."

    mock_rag = MagicMock()
    from src.schemas.responses import RuleCitation

    mock_rag.get_indexed_tomes.return_value = ["PHB.pdf"]
    mock_rag.retrieve_relevant_rules.return_value = [
        RuleCitation(
        source="PHB.pdf",
        section="Combat > Grappling",
        page=195,
        text="Grappling involves an Athletics check.",
        )
    ]

    service = AIService(model_client=mock_client, rag=mock_rag)

    history = [
        {
            "user": "How does grappling work?",
            "assistant": "It involves an Athletics check.",
        }
    ]

    service.process_message(
        "What about escaping?",
        history=history,
    )

    prompt_sent = mock_client.generate.call_args.args[0]
    assert "How does grappling work?" in prompt_sent
    assert "What about escaping?" in prompt_sent
    # Pattern B excludes prior assistant answers to avoid self-poisoning
    assert "It involves an Athletics check." not in prompt_sent


def test_conversation_history_is_included_without_rule_excerpts():
    """Verify history is included when no rule excerpts are available."""
    mock_client = MagicMock()
    mock_client.model_name = "test_model"
    mock_client.generate.return_value = "You can escape using an action."

    mock_rag = MagicMock()
    mock_rag.get_indexed_tomes.return_value = []
    mock_rag.retrieve_relevant_rules.return_value = []

    service = AIService(model_client=mock_client, rag=mock_rag)

    history = [
        {
            "user": "How does grappling work?",
            "assistant": "The target is grappled.",
        }
    ]

    service.process_message(
        "How do I escape?",
        history=history,
    )

    assert mock_client.generate.called
    prompt_sent = mock_client.generate.call_args.args[0]

    assert "How does grappling work?" in prompt_sent
    assert "How do I escape?" in prompt_sent
    # Pattern B excludes prior assistant answers to avoid self-poisoning
    assert "The target is grappled." not in prompt_sent



def test_conversation_history_is_used_for_rag_retrieval():
    """Verify prior user questions provide context to RAG retrieval."""
    mock_client = MagicMock()
    mock_client.model_name = "test_model"
    mock_client.generate.return_value = "You can escape the grapple."

    mock_rag = MagicMock()
    mock_rag.get_indexed_tomes.return_value = []
    mock_rag.retrieve_relevant_rules.return_value = []

    service = AIService(model_client=mock_client, rag=mock_rag)

    history = [
        {
            "user": "How does grappling work?",
            "assistant": "The target is grappled.",
        }
    ]

    service.process_message(
        "What about escaping?",
        history=history,
    )

    retrieval_query = mock_rag.retrieve_relevant_rules.call_args.args[0]

    assert "How does grappling work?" in retrieval_query
    assert "What about escaping?" in retrieval_query

def test_model_client_system_prompt_handling():
    """Verify that OllamaModelClient properly passes native system messages when system prompt is provided."""
    from unittest.mock import patch
    from src.models.model_client import OllamaModelClient

    with patch("ollama.Client") as mock_ollama_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {"message": {"content": "Verdict"}}
        mock_ollama_client_cls.return_value = mock_instance

        client = OllamaModelClient(model_name="test_model")
        client.generate("User query", system="System directive")

        mock_instance.chat.assert_called_with(
            model="test_model",
            messages=[
                {"role": "system", "content": "System directive"},
                {"role": "user", "content": "User query"},
            ],
            options={"temperature": config.model_temperature},
        )


def test_rag_pipeline_uses_configured_top_k():
    """Verify that RAGPipeline defaults to config.rag_top_k when n_results is omitted."""
    from src.capabilities.rag import RAGPipeline
    from unittest.mock import patch

    with patch.object(RAGPipeline, "_get_collection") as mock_get_coll:
        mock_coll = MagicMock()
        mock_coll.count.return_value = 10
        mock_coll.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_get_coll.return_value = mock_coll

        rag = RAGPipeline()
        rag.retrieve_relevant_rules("test query")

        mock_coll.query.assert_called_once()
        _, kwargs = mock_coll.query.call_args
        assert kwargs["n_results"] == config.rag_top_k


def test_rag_pipeline_bm25_rrf_reranking():
    """Verify that BM25 RRF re-ranks vector candidates to boost exact keyword terminology matches."""
    from src.capabilities.rag import RAGPipeline
    from unittest.mock import patch

    pipeline = RAGPipeline()
    mock_coll = MagicMock()
    mock_coll.count.return_value = 3

    # Vector query returns Doc 1 first (lower distance), Doc 2 second
    mock_coll.query.return_value = {
        "documents": [[
            "A character can take various general maneuvers during an encounter.",
            "Cunning Action allows you to Dash, Disengage, or Hide as a bonus action.",
        ]],
        "metadatas": [[
            {"source": "Core.pdf", "section": "Combat", "page": 10},
            {"source": "Core.pdf", "section": "Rogue Features", "page": 20},
        ]],
        "distances": [[0.60, 0.75]],
        "ids": [["doc_1", "doc_2"]],
    }

    # Document store contains corpus (N >= 3 ensures positive IDF in BM25Okapi)
    mock_coll.get.return_value = {
        "documents": [
            "A character can take various general maneuvers during an encounter.",
            "Cunning Action allows you to Dash, Disengage, or Hide as a bonus action.",
            "Spellcasters prepare magical spells from their spellbooks each dawn.",
        ],
        "ids": ["doc_1", "doc_2", "doc_3"],
    }

    with patch.object(pipeline, "_get_collection", return_value=mock_coll):
        # Specific keyword query targeting Cunning Action
        results = pipeline.retrieve_relevant_rules("How does Cunning Action work?", selected_tomes=["Core.pdf"])
        assert len(results) == 2
        # Doc 2 should be boosted to rank 1 via BM25 RRF because of exact terminology match
        assert results[0].section == "Rogue Features"
        assert "Cunning Action" in results[0].text
        assert results[1].section == "Combat"


def test_rag_pipeline_bm25_cache_invalidation():
    """Verify that deleting a rulebook clears the BM25 cache."""
    from src.capabilities.rag import RAGPipeline

    pipeline = RAGPipeline()
    pipeline._bm25_cache["Core.pdf"] = ("mock_bm25", ["id1"])
    mock_coll = MagicMock()
    pipeline._collection = mock_coll

    assert len(pipeline._bm25_cache) > 0
    pipeline.delete_rulebook("Core.pdf")
    assert len(pipeline._bm25_cache) == 0



