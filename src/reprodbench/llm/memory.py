from typing import Dict

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import BaseMessage

_MAX_MESSAGES = 5
_MESSAGE_STORE: Dict[str, BaseChatMessageHistory] = {}


class WindowedChatMessageHistory(InMemoryChatMessageHistory):
    """In-memory chat history with a fixed message window."""

    def add_message(self, message: BaseMessage) -> None:
        super().add_message(message)

        # Enforce sliding window
        if len(self.messages) > _MAX_MESSAGES:
            self.messages = self.messages[-_MAX_MESSAGES:]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _MESSAGE_STORE:
        _MESSAGE_STORE[session_id] = WindowedChatMessageHistory()
    return _MESSAGE_STORE[session_id]


def print_session_history(session_id: str) -> None:
    history = get_session_history(session_id)
    for i, msg in enumerate(history.messages):
        print(f"[{i}] {msg.type.upper()}:")
        print(msg.content)
        print("-" * 80)
