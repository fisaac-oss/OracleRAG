import re
from typing import Optional
import ollama
from src.config import config


class GenerationResult(str):
    """String subclass that carries optional thinking metadata while maintaining full str compatibility."""

    thinking: Optional[str] = None

    def __new__(cls, content: str, thinking: Optional[str] = None):
        instance = super().__new__(cls, content)
        instance.thinking = thinking
        return instance


class ModelClientError(Exception):
    """Base exception for model client failures."""
    pass


class OllamaConnectionError(ModelClientError):
    """Exception raised when Ollama service is unreachable."""
    pass


class ModelNotFoundError(ModelClientError):
    """Exception raised when the requested model is not found on local Ollama."""
    pass


def _extract_draft_verdict(thinking: str) -> str:
    """Attempts to extract a meaningful draft answer or conclusion from an uncompleted thinking trace."""
    lines = [line.strip() for line in thinking.split("\n") if line.strip()]
    if not lines:
        return ""
    # Look for draft or conclusion indicators
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(marker in line_lower for marker in ["draft response", "draft answer", "conclusion:", "in summary", "final answer"]):
            candidate = "\n".join(lines[i:])
            if len(candidate) > 20:
                return candidate
    # Fallback to the last few non-empty lines
    return "\n".join(lines[-4:])


class OllamaModelClient:
    """Encapsulates interaction with local Ollama inference server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.base_url = base_url or config.ollama_base_url
        self.model_name = model_name or config.model_name
        self.temperature = (
            temperature if temperature is not None else config.model_temperature
        )
        self._client = ollama.Client(host=self.base_url)

    def list_available_models(self) -> list[str]:
        """Returns a sorted list of model tags installed in local Ollama."""
        try:
            res = self._client.list()
            raw_models = getattr(res, "models", []) or (
                res.get("models", []) if isinstance(res, dict) else []
            )
            names = []
            for m in raw_models:
                m_name = (
                    getattr(m, "model", None)
                    or getattr(m, "name", None)
                    or (m.get("model") if isinstance(m, dict) else None)
                )
                if m_name:
                    names.append(m_name)
            return sorted(list(set(names)))
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model_name: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> GenerationResult:
        """
        Sends a user prompt to local Ollama model and returns a GenerationResult string.
        Extracts native thinking/deliberation traces from thinking models (e.g. qwen3.5:4b, deepseek-r1).
        Supports explicitly toggling thinking mode on or off via the think parameter.
        
        Raises ModelClientError subclass if connection or execution fails.
        """
        target_model = model_name or self.model_name
        is_thinking = any(
            k in target_model.lower() for k in ["qwen3.5", "r1", "qwq", "reason"]
        )

        temp = self.temperature if temperature is None else temperature
        # Thinking models degrade or loop when temperature is close to 0.0
        if is_thinking and (temp is None or temp <= 0.2):
            temp = 0.6

        options = {"temperature": temp}
        if is_thinking and think is not False:
            # Thinking models generate extensive internal reasoning before the final verdict.
            # Provide sufficient output budget and context window so deliberation doesn't truncate the answer.
            options["num_predict"] = 3072
            options["num_ctx"] = 8192

        # Format messages, separating system instructions if present
        if "--- SACRED RULE EXCERPTS ---" in prompt and "You are an impartial" in prompt:
            parts = prompt.split("--- SACRED RULE EXCERPTS ---", 1)
            messages = [
                {"role": "system", "content": parts[0].strip()},
                {"role": "user", "content": f"--- SACRED RULE EXCERPTS ---{parts[1]}"},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        chat_kwargs = {
            "model": target_model,
            "messages": messages,
            "options": options,
        }
        if think is not None:
            chat_kwargs["think"] = think

        try:
            response = self._client.chat(**chat_kwargs)

            content = ""
            thinking: Optional[str] = None

            # Standardize extraction from dict or chat response object
            if isinstance(response, dict):
                msg = response.get("message", {})
                content = msg.get("content", "")
                thinking = msg.get("thinking")
            elif hasattr(response, "message"):
                msg = response.message
                content = getattr(msg, "content", "")
                thinking = getattr(msg, "thinking", None)
            else:
                content = str(response)

            # Fallback parsing: if thinking wasn't captured in a distinct field,
            # parse <think>...</think> tags if embedded directly in content
            if not thinking and isinstance(content, str):
                if "<think>" in content and "</think>" in content:
                    match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
                    if match:
                        thinking = match.group(1).strip()
                        content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        ).strip()
                elif "<think>" in content and "</think>" not in content:
                    parts = content.split("<think>", 1)
                    content = parts[0].strip()
                    thinking = parts[1].strip()

            if thinking is not None:
                thinking = str(thinking).strip() or None

            # Auto-recovery if reasoning consumed the output budget before drafting content
            if not content.strip() and thinking:
                # Attempt immediate recovery in direct mode (think=False) so user receives the full decree
                try:
                    recovery_options = dict(options)
                    recovery_options["num_predict"] = 1536
                    rec_response = self._client.chat(
                        model=target_model,
                        messages=messages,
                        options=recovery_options,
                        think=False,
                    )
                    rec_content = ""
                    if isinstance(rec_response, dict):
                        rec_content = rec_response.get("message", {}).get("content", "")
                    elif hasattr(rec_response, "message"):
                        rec_content = getattr(rec_response.message, "content", "")
                    else:
                        rec_content = str(rec_response)

                    if rec_content and rec_content.strip():
                        content = rec_content.strip()
                except Exception:
                    pass

                # If recovery was not possible, surface preliminary insight from thoughts
                if not content.strip():
                    draft_insight = _extract_draft_verdict(thinking)
                    draft_section = (
                        f"\n\n**Preliminary Insight from Deliberation:**\n{draft_insight}"
                        if draft_insight
                        else ""
                    )
                    content = (
                        "⚠️ *The Oracle's contemplation reached its maximum token capacity before finalizing the formal decree.*"
                        f"{draft_section}\n\n"
                        "*(Expand **Oracle's Deliberation** above to review the full thought process, or toggle off Deep Reasoning for an immediate direct response.)*"
                    )

            return GenerationResult(content, thinking=thinking)

        except Exception as err:
            err_str = str(err).lower()
            if "connection" in err_str or "connect" in err_str or "refused" in err_str:
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama service at {self.base_url}."
                ) from err
            elif "not found" in err_str or "404" in err_str:
                raise ModelNotFoundError(
                    f"Model '{target_model}' is not installed in local Ollama."
                ) from err
            else:
                raise ModelClientError(f"Error communicating with Ollama: {err}") from err
