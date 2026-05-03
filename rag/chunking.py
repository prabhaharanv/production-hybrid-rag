"""Configurable chunking strategies using the Strategy pattern.

Strategies:
- word: Fixed-size word-level chunks with overlap (default, fast)
- sentence: Sentence-boundary-aware chunking (respects semantic units)
- recursive: Hierarchical splitting (paragraphs → sentences → words)
- token: Token-aware chunking using tiktoken byte-pair encoding

Set CHUNKING_STRATEGY env var or pass strategy= to chunk_documents().
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod


# ---- Strategy interface ----

class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        ...


# ---- Concrete strategies ----

class WordChunking(ChunkingStrategy):
    """Fixed-size word-level chunks with overlap."""

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-boundary-aware chunking. Groups sentences until chunk_size words."""

    _SENT_RE = re.compile(r'(?<=[.!?])\s+')

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        sentences = self._SENT_RE.split(text.strip())
        if not sentences:
            return []

        chunks = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > chunk_size and current:
                chunks.append(" ".join(current))
                # Overlap: keep last sentences that fit within overlap words
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    s_len = len(s.split())
                    if overlap_len + s_len > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += s_len
                current = overlap_sents
                current_len = overlap_len
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(" ".join(current))
        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Hierarchical splitting: paragraphs → sentences → words.

    Tries to split on paragraph boundaries first, falls back to sentences,
    then words. Produces chunks that respect document structure.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        return self._split(text, chunk_size, overlap, 0)

    def _split(self, text: str, chunk_size: int, overlap: int, sep_idx: int) -> list[str]:
        if len(text.split()) <= chunk_size:
            return [text.strip()] if text.strip() else []

        if sep_idx >= len(self._SEPARATORS):
            # Fallback: word-level split
            return WordChunking().chunk(text, chunk_size, overlap)

        sep = self._SEPARATORS[sep_idx]
        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate.split()) > chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                # Recursively split the part that's too large
                if len(part.split()) > chunk_size:
                    chunks.extend(self._split(part, chunk_size, overlap, sep_idx + 1))
                else:
                    current = part
                    continue
                current = ""
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())
        return chunks


class TokenChunking(ChunkingStrategy):
    """Token-aware chunking using tiktoken BPE tokenizer.

    chunk_size and overlap are interpreted as token counts.
    """

    def __init__(self):
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise ImportError("tiktoken is required for token chunking: pip install tiktoken")

    def chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(self._enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start = max(end - overlap, start + 1)
        return chunks


# ---- Strategy registry ----

STRATEGIES: dict[str, type[ChunkingStrategy]] = {
    "word": WordChunking,
    "sentence": SentenceChunking,
    "recursive": RecursiveChunking,
    "token": TokenChunking,
}


def get_chunking_strategy(name: str | None = None) -> ChunkingStrategy:
    """Get a chunking strategy by name. Defaults to CHUNKING_STRATEGY env var or 'word'."""
    strategy_name = name or os.getenv("CHUNKING_STRATEGY", "word")
    cls = STRATEGIES.get(strategy_name)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: '{strategy_name}'. Available: {list(STRATEGIES.keys())}")
    return cls()


# ---- Public API (backward compatible) ----

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60, strategy: str | None = None) -> list[str]:
    s = get_chunking_strategy(strategy)
    return s.chunk(text, chunk_size, overlap)


def chunk_documents(documents: list[dict], chunk_size: int = 400, overlap: int = 60, strategy: str | None = None) -> list[dict]:
    s = get_chunking_strategy(strategy)
    chunked = []

    for doc in documents:
        chunks = s.chunk(doc["text"], chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            chunked.append(
                {
                    "chunk_id": f'{doc["doc_id"]}_chunk_{idx}',
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "text": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": idx,
                        "chunking_strategy": s.__class__.__name__,
                    },
                }
            )

    return chunked