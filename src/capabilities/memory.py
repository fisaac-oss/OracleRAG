
from collections import deque


class ConversationMemory:
    """Keep a limited number of recent user/assistant turns."""

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self._history = deque(maxlen=max_turns)

    def add_turn(self, user_message: str, assistant_response: str) -> None:
        self._history.append({
            "user": user_message,
            "assistant": assistant_response,
        })

    def get_history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
