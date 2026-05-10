"""Tests for contextual compression."""

import numpy as np
from unittest.mock import MagicMock, patch

from rag.compressor import LLMCompressor, EmbeddingCompressor


class TestLLMCompressor:
    def test_compress_filters_irrelevant(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="RAG combines retrieval and generation.")
                )
            ]
        )

        with patch("rag.compressor.OpenAI", return_value=mock_client):
            compressor = LLMCompressor(model="test", api_key="key")

        compressor.client = mock_client
        chunks = [
            {
                "text": "RAG combines retrieval and generation. The weather is sunny. Dogs are great.",
                "metadata": {},
            },
        ]
        result = compressor.compress("What is RAG?", chunks)

        assert len(result) == 1
        assert result[0]["text"] == "RAG combines retrieval and generation."
        assert result[0]["metadata"]["compressed"] is True
        assert (
            result[0]["metadata"]["original_length"]
            > result[0]["metadata"]["compressed_length"]
        )

    def test_compress_returns_originals_when_all_filtered(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="NONE"))]
        )

        with patch("rag.compressor.OpenAI", return_value=mock_client):
            compressor = LLMCompressor(model="test", api_key="key")

        compressor.client = mock_client
        chunks = [{"text": "Irrelevant content.", "metadata": {}}]
        result = compressor.compress("query", chunks)

        # Should fallback to originals
        assert result == chunks

    def test_compress_multiple_chunks(self):
        mock_client = MagicMock()
        responses = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Relevant sentence 1."))]
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Relevant sentence 2."))]
            ),
        ]
        mock_client.chat.completions.create.side_effect = responses

        with patch("rag.compressor.OpenAI", return_value=mock_client):
            compressor = LLMCompressor(model="test", api_key="key")

        compressor.client = mock_client
        chunks = [
            {
                "text": "First chunk with several sentences. Some irrelevant.",
                "metadata": {},
            },
            {"text": "Second chunk. Also some noise here.", "metadata": {}},
        ]
        result = compressor.compress("query", chunks)
        assert len(result) == 2
        assert all(r["metadata"]["compressed"] for r in result)


class FakeEmbedder:
    """Returns embeddings that make some sentences more similar to query."""

    def embed_query(self, text: str) -> np.ndarray:
        # Query gets a fixed embedding
        return np.array([[1.0, 0.0, 0.0]])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            if "RAG" in text or "retrieval" in text:
                # High similarity to query
                embeddings.append([0.9, 0.1, 0.0])
            else:
                # Low similarity
                embeddings.append([0.1, 0.9, 0.0])
        return np.array(embeddings)


class TestEmbeddingCompressor:
    def setup_method(self):
        self.embedder = FakeEmbedder()
        self.compressor = EmbeddingCompressor(
            embedder=self.embedder,
            similarity_threshold=0.5,
            min_sentences=1,
        )

    def test_filters_irrelevant_sentences(self):
        chunks = [
            {
                "text": "RAG combines retrieval and generation. The weather is nice today. Dogs are loyal pets.",
                "metadata": {},
            }
        ]
        result = self.compressor.compress("What is RAG?", chunks)
        assert len(result) == 1
        # Should keep RAG-related sentence, filter others
        assert "RAG" in result[0]["text"]
        assert result[0]["metadata"]["compressed"] is True

    def test_keeps_min_sentences(self):
        compressor = EmbeddingCompressor(
            embedder=self.embedder,
            similarity_threshold=0.99,  # Very high threshold
            min_sentences=2,
        )
        chunks = [
            {
                "text": "First sentence here. Second sentence there. Third one.",
                "metadata": {},
            }
        ]
        result = compressor.compress("query", chunks)
        assert result[0]["metadata"]["sentences_kept"] >= 2

    def test_passes_through_short_chunks(self):
        chunks = [{"text": "Single sentence.", "metadata": {}}]
        result = self.compressor.compress("query", chunks)
        # Single sentence chunks pass through unchanged
        assert result[0]["text"] == "Single sentence."

    def test_multiple_chunks(self):
        chunks = [
            {"text": "RAG is about retrieval. Cats are cute.", "metadata": {}},
            {"text": "Dense retrieval uses embeddings. Pizza is food.", "metadata": {}},
        ]
        result = self.compressor.compress("What is retrieval?", chunks)
        assert len(result) == 2
        # Both should have retrieval-related content emphasized
        for r in result:
            assert r["metadata"]["compressed"] is True
