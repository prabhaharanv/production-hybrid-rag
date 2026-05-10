"""Tests for adaptive retrieval routing."""

from unittest.mock import MagicMock

from rag.adaptive import QueryComplexityClassifier, AdaptiveRetriever


class TestQueryComplexityClassifier:
    def setup_method(self):
        self.classifier = QueryComplexityClassifier()

    def test_simple_short_query(self):
        assert self.classifier.classify("What is RAG?") == "simple"

    def test_simple_define(self):
        assert self.classifier.classify("define chunking") == "simple"

    def test_simple_single_word(self):
        assert self.classifier.classify("embeddings?") == "simple"

    def test_complex_comparison(self):
        result = self.classifier.classify(
            "Compare dense retrieval versus sparse retrieval and explain the trade-offs"
        )
        assert result == "complex"

    def test_complex_multi_hop(self):
        result = self.classifier.classify(
            "How does chunking strategy affect retrieval quality and what is the relationship between chunk size and recall?"
        )
        assert result == "complex"

    def test_moderate_medium_length(self):
        result = self.classifier.classify(
            "What are the main components of a retrieval augmented generation pipeline?"
        )
        assert result in ("moderate", "complex")  # Acceptable for medium queries

    def test_very_short_always_simple(self):
        assert self.classifier.classify("RAG") == "simple"
        assert self.classifier.classify("hello world") == "simple"


class TestAdaptiveRetriever:
    def setup_method(self):
        self.dense = MagicMock()
        self.sparse = MagicMock()
        self.hybrid = MagicMock()

        self.dense.retrieve.return_value = [{"text": "dense result", "score": 0.9}]
        self.sparse.retrieve.return_value = [{"text": "sparse result", "score": 0.8}]
        self.hybrid.retrieve.return_value = [{"text": "hybrid result", "score": 0.95}]

    def test_simple_routes_to_sparse(self):
        retriever = AdaptiveRetriever(self.dense, self.sparse, self.hybrid)
        results = retriever.retrieve("What is RAG?", top_k=3)

        self.sparse.retrieve.assert_called_once_with("What is RAG?", top_k=3)
        self.dense.retrieve.assert_not_called()
        self.hybrid.retrieve.assert_not_called()
        assert results[0]["retrieval_strategy"] == "sparse"
        assert results[0]["query_complexity"] == "simple"

    def test_complex_routes_to_hybrid(self):
        retriever = AdaptiveRetriever(self.dense, self.sparse, self.hybrid)
        query = "Compare the trade-offs between dense versus sparse retrieval approaches"
        results = retriever.retrieve(query, top_k=5)

        self.hybrid.retrieve.assert_called_once()
        assert results[0]["retrieval_strategy"] == "hybrid"
        assert results[0]["query_complexity"] == "complex"

    def test_moderate_routes_to_dense(self):
        # Force moderate classification
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = "moderate"

        retriever = AdaptiveRetriever(self.dense, self.sparse, self.hybrid, classifier=mock_classifier)
        results = retriever.retrieve("something", top_k=5)

        self.dense.retrieve.assert_called_once()
        assert results[0]["retrieval_strategy"] == "dense"

    def test_custom_classifier(self):
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = "complex"

        retriever = AdaptiveRetriever(self.dense, self.sparse, self.hybrid, classifier=mock_classifier)
        retriever.retrieve("any query", top_k=3)

        mock_classifier.classify.assert_called_once_with("any query")
        self.hybrid.retrieve.assert_called_once()

    def test_top_k_passed_through(self):
        retriever = AdaptiveRetriever(self.dense, self.sparse, self.hybrid)
        retriever.retrieve("test", top_k=10)
        self.sparse.retrieve.assert_called_once_with("test", top_k=10)
