"""Conversation memory for multi-turn context.

Maintains a sliding window of previous turns (question + answer pairs)
and injects them into the prompt so the LLM can resolve references
like "it", "that", "the previous one", etc.

Memory is per-session (keyed by conversation_id). Old turns are evicted
when the window exceeds ``max_turns``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Turn:
    """A single conversation turn."""

    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Thread-safe sliding-window conversation memory.

    Parameters
    ----------
    max_turns : int
        Maximum number of past turns to keep per conversation.
    ttl : int
        Seconds after which an idle conversation is evicted (0 = never).
    """

    def __init__(self, max_turns: int = 5, ttl: int = 3600):
        self.max_turns = max_turns
        self.ttl = ttl
        self._conversations: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def add_turn(self, conversation_id: str, question: str, answer: str) -> None:
        """Record a completed turn."""
        with self._lock:
            turns = self._conversations.setdefault(conversation_id, [])
            turns.append(Turn(question=question, answer=answer))
            if len(turns) > self.max_turns:
                self._conversations[conversation_id] = turns[-self.max_turns :]

    def get_history(self, conversation_id: str) -> list[Turn]:
        """Return the conversation history (oldest-first)."""
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def clear(self, conversation_id: str) -> None:
        """Delete a conversation's history."""
        with self._lock:
            self._conversations.pop(conversation_id, None)

    def clear_all(self) -> None:
        """Delete all conversations."""
        with self._lock:
            self._conversations.clear()

    def list_conversations(self) -> list[dict]:
        """Return metadata for all active conversations."""
        with self._lock:
            result = []
            for cid, turns in self._conversations.items():
                result.append(
                    {
                        "conversation_id": cid,
                        "turn_count": len(turns),
                        "last_activity": turns[-1].timestamp if turns else 0,
                    }
                )
            return result

    def evict_stale(self) -> int:
        """Remove conversations idle longer than TTL. Returns count evicted."""
        if self.ttl <= 0:
            return 0
        cutoff = time.time() - self.ttl
        evicted = 0
        with self._lock:
            stale = [
                cid
                for cid, turns in self._conversations.items()
                if turns and turns[-1].timestamp < cutoff
            ]
            for cid in stale:
                del self._conversations[cid]
                evicted += 1
        return evicted


def build_history_context(turns: list[Turn]) -> str:
    """Format conversation history for injection into the RAG prompt."""
    if not turns:
        return ""
    lines = ["Previous conversation:"]
    for turn in turns:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    lines.append("")
    return "\n".join(lines)
