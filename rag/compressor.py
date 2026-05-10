"""Contextual compression: extract only relevant sentences from retrieved chunks.

Before sending retrieved chunks to the LLM for final generation, this module
uses an LLM call to extract only the sentences that are directly relevant to
the query. This reduces noise and improves answer quality while staying within
context window limits.

Two approaches:
1. LLMCompressor: Uses the LLM to extract relevant sentences (higher quality)
2. EmbeddingCompressor: Uses embedding similarity to filter sentences (faster, no LLM call)
"""

from __future__ import annotations

import re

import numpy as np
from openai import OpenAI


class LLMCompressor:
    """Uses LLM to extract only relevant sentences from a chunk."""

    COMPRESSION_PROMPT = (
        "Given the following context and question, extract ONLY the sentences "
        "from the context that are directly relevant to answering the question. "
        "Return the relevant sentences verbatim, one per line. "
        "If no sentences are relevant, return 'NONE'.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Relevant sentences:"
    )

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def compress(self, query: str, chunks: list[dict]) -> list[dict]:
        """Compress chunks to only relevant sentences."""
        compressed = []

        for chunk in chunks:
            relevant_text = self._extract_relevant(query, chunk["text"])
            if relevant_text:
                compressed.append({
                    **chunk,
                    "text": relevant_text,
                    "metadata": {
                        **chunk.get("metadata", {}),
                        "compressed": True,
                        "original_length": len(chunk["text"]),
                        "compressed_length": len(relevant_text),
                    },
                })

        return compressed if compressed else chunks  # Fallback to originals if all filtered

    def _extract_relevant(self, query: str, context: str) -> str | None:
        prompt = self.COMPRESSION_PROMPT.format(question=query, context=context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You extract relevant sentences from documents."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        result = response.choices[0].message.content.strip()
        if result.upper() == "NONE" or not result:
            return None
        return result


class EmbeddingCompressor:
    """Uses embedding similarity to filter sentences — faster, no LLM call.

    Splits each chunk into sentences, embeds them alongside the query,
    and keeps only sentences above a similarity threshold.
    """

    _SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, embedder, similarity_threshold: float = 0.3, min_sentences: int = 1):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences

    def compress(self, query: str, chunks: list[dict]) -> list[dict]:
        """Filter sentences in each chunk by embedding similarity to query."""
        query_embedding = self.embedder.embed_query(query).flatten()
        compressed = []

        for chunk in chunks:
            sentences = self._split_sentences(chunk["text"])
            if len(sentences) <= self.min_sentences:
                compressed.append(chunk)
                continue

            # Embed all sentences
            sentence_embeddings = self.embedder.embed_documents(sentences)

            # Compute similarities
            similarities = np.dot(sentence_embeddings, query_embedding)

            # Keep sentences above threshold
            relevant_indices = [
                i for i, sim in enumerate(similarities)
                if sim >= self.similarity_threshold
            ]

            # Ensure at least min_sentences are kept
            if len(relevant_indices) < self.min_sentences:
                top_indices = np.argsort(similarities)[-self.min_sentences:]
                relevant_indices = sorted(top_indices.tolist())

            relevant_text = " ".join(sentences[i] for i in sorted(relevant_indices))

            compressed.append({
                **chunk,
                "text": relevant_text,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "compressed": True,
                    "original_length": len(chunk["text"]),
                    "compressed_length": len(relevant_text),
                    "sentences_kept": len(relevant_indices),
                    "sentences_total": len(sentences),
                },
            })

        return compressed

    def _split_sentences(self, text: str) -> list[str]:
        sentences = self._SENTENCE_RE.split(text.strip())
        return [s for s in sentences if s.strip()]
