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
    assert "SACRED RULE EXCERPTS" in generated_prompt
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
