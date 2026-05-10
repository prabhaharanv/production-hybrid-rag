"""Tests for HyDE (Hypothetical Document Embeddings) retrieval."""

import numpy as np
from unittest.mock import MagicMock, patch

from rag.hyde import HyDEGenerator, HyDERetriever, HYDE_PROMPT


class TestHyDEGenerator:
    def test_generate_hypothetical(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="RAG retrieval combines dense and sparse methods."))]
        )

        with patch("rag.hyde.OpenAI", return_value=mock_client):
            gen = HyDEGenerator(model="test-model", api_key="key", base_url="http://localhost")

        gen.client = mock_client
        result = gen.generate_hypothetical("What is RAG?")
        assert result == "RAG retrieval combines dense and sparse methods."
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 150

    def test_prompt_contains_question(self):
        formatted = HYDE_PROMPT.format(question="How does chunking work?")
        assert "How does chunking work?" in formatted


class TestHyDERetriever:
    def test_retrieve_uses_hypothetical_doc_embedding(self):
        mock_hyde_gen = MagicMock()
        mock_hyde_gen.generate_hypothetical.return_value = "A hypothetical answer about RAG."

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {"chunk_id": "c1", "text": "Real document chunk", "score": 0.95},
        ]

        retriever = HyDERetriever(mock_hyde_gen, mock_embedder, mock_vector_store)
        results = retriever.retrieve("What is RAG?", top_k=3)

        # Verifies the flow: generate hypothetical → embed it → search
        mock_hyde_gen.generate_hypothetical.assert_called_once_with("What is RAG?")
        mock_embedder.embed_query.assert_called_once_with("A hypothetical answer about RAG.")
        mock_vector_store.search.assert_called_once()

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["hyde_doc"] == "A hypothetical answer about RAG."

    def test_retrieve_passes_top_k(self):
        mock_hyde_gen = MagicMock()
        mock_hyde_gen.generate_hypothetical.return_value = "Hypothetical."
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.array([[0.5]])
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []

        retriever = HyDERetriever(mock_hyde_gen, mock_embedder, mock_vector_store)
        retriever.retrieve("test", top_k=10)

        _, kwargs = mock_vector_store.search.call_args
        assert kwargs["top_k"] == 10
