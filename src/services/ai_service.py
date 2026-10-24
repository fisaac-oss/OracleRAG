import time
from typing import Optional, BinaryIO, Callable
from src.models.model_client import (
    OllamaModelClient,
    ModelClientError,
    OllamaConnectionError,
    ModelNotFoundError,
)
from src.schemas.responses import (
    UserRequest,
    AIResponse,
    IngestionResponse,
    RuleCitation,
)
from src.capabilities.rag import get_rag_pipeline, RAGPipeline
from src.config import config


class AIService:
    """
    Application service layer responsible for validating user input,
    orchestrating model client requests, managing RAG document ingestion,
    and catching exceptions gracefully.
    """

    def __init__(
        self,
        model_client: Optional[OllamaModelClient] = None,
        rag: Optional[RAGPipeline] = None,
    ):
        # Allow injecting custom/mock model_client and rag for simple testing
        self.model_client = model_client
        self.rag = rag

    def _get_client(self) -> OllamaModelClient:
        """Returns active model client, initializing default client if none provided."""
        if self.model_client is None:
            self.model_client = OllamaModelClient()
        return self.model_client

    def _get_rag(self) -> RAGPipeline:
        """Returns active RAG pipeline, initializing default pipeline if none provided."""
        if self.rag is None:
            self.rag = get_rag_pipeline()
        return self.rag

    def ingest_rulebook(
        self,
        file_source: str | bytes | BinaryIO,
        filename: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> IngestionResponse:
        """
        Ingests and indexes an uploaded rulebook into the RAG vector store.
        """
        try:
            rag = self._get_rag()
            if progress_callback is not None:
                count = rag.ingest_document(
                    file_source,
                    filename,
                    progress_callback=progress_callback,
                )
            else:
                count = rag.ingest_document(file_source, filename)
            return IngestionResponse(
                filename=filename,
                chunks_count=count,
                success=True,
            )
        except Exception as err:
            return IngestionResponse(
                filename=filename,
                chunks_count=0,
                success=False,
                error_message=str(err),
            )

    def get_indexed_tomes(self) -> list[str]:
        """Returns the list of currently indexed rulebooks."""
        try:
            return self._get_rag().get_indexed_tomes()
        except Exception:
            return []

    def delete_rulebook(self, filename: str) -> bool:
        """Deletes a rulebook and all its chunks from persistent ChromaDB vector storage."""
        try:
            return self._get_rag().delete_rulebook(filename)
        except Exception:
            return False

    
    def process_message(
        self,
        user_message: str,
        selected_tomes: Optional[list[str]] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> AIResponse:

        """
        Processes a raw user message string and returns a structured AIResponse.
        Catches technical failures and converts them to friendly user-facing messages.
        Optionally limits RAG retrieval to selected_tomes.
        """
        start_time = time.perf_counter()

        # 1. Validate empty input
        if not user_message or not user_message.strip():
            return AIResponse(
                content="Please enter a message before sending.",
                success=False,
                error_message="User message was empty.",
                elapsed_time=round(time.perf_counter() - start_time, 2),
            )

        try:
            # 2. Schema validation
            request = UserRequest(
                message=user_message.strip(),
                selected_tomes=selected_tomes,
            )

            
            # Format the last three user queries for reference resolution (Pattern B: Question-only)
            recent_history = (history or [])[-3:]
            recent_user_questions = [
                turn.get("user", "").strip()
                for turn in recent_history
                if turn.get("user", "").strip()
            ]
            history_context = "\n".join(f"- {q}" for q in recent_user_questions)

            # 3. Retrieve relevant rulebook excerpts via RAG
            rag = self._get_rag()

            retrieval_query = request.message
            if recent_user_questions:
                retrieval_query = (
                    "\n".join(recent_user_questions)
                    + "\n"
                    + request.message
                )

            citations: list[RuleCitation] = rag.retrieve_relevant_rules(
                retrieval_query,
                selected_tomes=request.selected_tomes,
            )


            indexed_tomes = rag.get_indexed_tomes()

            # 4. Call model client
            client = self._get_client()
            model_name = (
                client.model_name
                if isinstance(getattr(client, "model_name", None), str)
                else "oracle_model"
            )

            # Fallback Guardrail 1: If tomes are indexed, but the user explicitly selected no tomes
            if (
                indexed_tomes
                and request.selected_tomes is not None
                and len(request.selected_tomes) == 0
            ):
                return AIResponse(
                    content=(
                        "📜 *No grimoires or scriptures have been selected for consultation.*\n\n"
                        "Please select at least one tome from the library to divine answers from."
                    ),
                    success=True,
                    model=model_name,
                    citations=[],
                    elapsed_time=round(time.perf_counter() - start_time, 2),
                )

            # Fallback Guardrail 2: If tomes are indexed/selected, but no rules matched the threshold
            if (indexed_tomes or request.selected_tomes) and not citations:
                tome_context = (
                    "selected tomes"
                    if request.selected_tomes is not None
                    else "currently indexed tomes"
                )
                return AIResponse(
                    content=(
                        "📜 *The consulted scriptures contain no decrees regarding this matter.*\n\n"
                        f"The Oracle cannot divine an answer based on your {tome_context}."
                    ),
                    success=True,
                    model=model_name,
                    citations=[],
                    elapsed_time=round(time.perf_counter() - start_time, 2),
                )

            # 4. Construct grounded prompt if rulebook excerpts were found
            if citations:
                context_blocks = "\n\n".join(
                    f"[{i+1}] Tome: {c.source} | Section: {c.section} | Page: {c.page or 'N/A'}\n{c.text}"
                    for i, c in enumerate(citations)
                )
                history_section = (
                    f"--- RECENT INQUIRIES (for reference context only) ---\n"
                    f"{history_context}\n\n"
                    if history_context
                    else ""
                )
                system_prompt = (
                    "You are an authoritative Tabletop RPG Rules Arbiter.\n"
                    "Answer user questions strictly and impartially using only the provided rulebook excerpts.\n\n"
                    "RULES:\n"
                    "1. Grounding: Rely ONLY on the excerpts provided. Do not use outside knowledge, invent house rules, "
                    "or assume rules from different systems or editions. Do not mistake section headers or OCR artifacts (e.g. 'Prerequisite:...') for in-game feature names.\n"
                    "2. Baseline vs. Optional Features: Always distinguish baseline combat actions (available to all characters) "
                    "from optional exceptions like Feats, Spells, or Class Features. State the universal baseline rule first. "
                    "Any excerpt with a 'Prerequisite:' represents an optional feat, NOT a universal rule. For example, the baseline 'grappled' condition "
                    "ONLY reduces the target's speed to 0; attack advantage and pinning are NOT part of standard grappling. "
                    "NEVER present benefits from feats, spells, or prerequisite-gated excerpts as default rules of a base action; "
                    "if mentioned, explicitly label them as requiring an optional feat.\n"
                    "3. Actor vs. Target Precision: Strictly distinguish who is affected by mechanics or penalties. "
                    "Do not confuse an initiator's penalty (e.g. a grappler's halved dragging speed) with a target's condition (e.g. a grappled target's speed reduced to 0).\n"
                    "4. Anti-Premise & Gates: Action prerequisites (such as size limits, reach, or conditions) are absolute gates. "
                    "If a query asks about an invalid action (e.g. grappling a target two sizes larger) or assumes non-existent mechanics "
                    "(e.g. death saving throws in Call of Cthulhu), state clearly that the action cannot be taken or the rule does not exist in the consulted scripture. "
                    "Never invent resolution steps, auto-successes, or escape checks for invalid actions.\n"
                    "5. Specificity: Distinguish strictly between concrete mechanics (dice rolls, stats, actions) and thematic lore. "
                    "Report only mechanics actually named in the passages. If an excerpt is partial, answer only what is supported.\n"
                    "6. Style: Write naturally and concisely (1-4 paragraphs), like a seasoned Game Master. "
                    "State the rule answer directly without meta-framing (e.g. do not say 'Based on the excerpts')."
                )

                prompt_to_model = (
                    f"--- RULE EXCERPTS ---\n{context_blocks}\n\n"
                    f"{history_section}"
                    f"--- USER QUESTION ---\n{request.message}\n\n"
                    "--- ARBITER'S VERDICT ---"
                )
                response_text = client.generate(prompt_to_model, system=system_prompt)

            else:
                # If no tomes are indexed yet, invoke base model with recent conversation context
                if history_context:
                    prompt_to_model = (
                        f"--- RECENT INQUIRIES (for reference context only) ---\n"
                        f"{history_context}\n\n"
                        f"--- USER QUESTION ---\n{request.message}"
                    )
                else:
                    prompt_to_model = request.message

                response_text = client.generate(prompt_to_model)

            return AIResponse(
                content=response_text,
                success=True,
                model=model_name,
                citations=citations,
                elapsed_time=round(time.perf_counter() - start_time, 2),
            )

        except OllamaConnectionError as err:
            return AIResponse(
                content=(
                    "[Error] Could not connect to Ollama.\n\n"
                    "Please verify that Ollama is installed and running locally on your machine."
                ),
                success=False,
                error_message=str(err),
                elapsed_time=round(time.perf_counter() - start_time, 2),
            )

        except ModelNotFoundError as err:
            return AIResponse(
                content=(
                    f"[Error] The configured AI model is unavailable in Ollama.\n\n"
                    f"Please verify your MODEL_NAME setting or run 'ollama run <model_name>'."
                ),
                success=False,
                error_message=str(err),
                elapsed_time=round(time.perf_counter() - start_time, 2),
            )

        except ModelClientError as err:
            return AIResponse(
                content="[Error] An unexpected communication error occurred with the AI model.",
                success=False,
                error_message=str(err),
                elapsed_time=round(time.perf_counter() - start_time, 2),
            )

        except Exception as err:
            return AIResponse(
                content="[Error] An unexpected application error occurred.",
                success=False,
                error_message=str(err),
                elapsed_time=round(time.perf_counter() - start_time, 2),
            )


def generate_response(
    user_message: str,
    selected_tomes: Optional[list[str]] = None,
    history: Optional[list[dict[str, str]]] = None,
    service: Optional[AIService] = None,
) -> AIResponse:
    """
    Main reusable service entry point used by the UI layer.

    Accepts user input message, optional selected_tomes list, passes it to the AI service,
    and returns the generated AIResponse (with content, model, and citations).
    """
    active_service = service or AIService()
    return active_service.process_message(
        user_message=user_message,
        selected_tomes=selected_tomes,
        history=history,
    )


def ingest_rulebook_file(
    file_source: str | bytes | BinaryIO,
    filename: str,
    service: Optional[AIService] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> IngestionResponse:
    """
    Service entry point for ingesting rulebook files from the UI layer.
    """
    active_service = service or AIService()
    return active_service.ingest_rulebook(
        file_source,
        filename,
        progress_callback=progress_callback,
    )


def get_active_tomes(service: Optional[AIService] = None) -> list[str]:
    """
    Service entry point for fetching indexed rulebooks.
    """
    active_service = service or AIService()
    return active_service.get_indexed_tomes()


def delete_rulebook_file(filename: str, service: Optional[AIService] = None) -> bool:
    """
    Service entry point for deleting a rulebook from ChromaDB vector store.
    """
    active_service = service or AIService()
    return active_service.delete_rulebook(filename)
