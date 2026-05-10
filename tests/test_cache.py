"""Tests for semantic caching."""

import time

import numpy as np
import pytest

from rag.cache import SemanticCache


class FakeEmbedder:
    """Returns predictable embeddings for testing."""

    def __init__(self):
        self._call_count = 0
        self._query_map = {}

    def embed_query(self, text: str) -> np.ndarray:
        # Return consistent embedding for same text
        if text not in self._query_map:
            # Use hash to generate a pseudo-embedding
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            self._query_map[text] = rng.randn(1, 384).astype("float32")
        return self._query_map[text]


class TestSemanticCache:
    def setup_method(self):
        self.embedder = FakeEmbedder()
        self.cache = SemanticCache(
            embedder=self.embedder,
            similarity_threshold=0.9,
            ttl_seconds=60,
        )

    def test_miss_on_empty_cache(self):
        result = self.cache.get("What is RAG?")
        assert result is None

    def test_hit_on_exact_query(self):
        # Store a result
        expected = {"answer": "RAG is retrieval augmented generation."}
        self.cache.put("What is RAG?", expected)

        # Same query should hit
        result = self.cache.get("What is RAG?")
        assert result is not None
        assert result["answer"] == expected["answer"]
        assert result["_cache_hit"] is True
        assert result["_cache_score"] >= 0.9

    def test_miss_on_different_query(self):
        self.cache.put("What is RAG?", {"answer": "RAG answer"})

        # Very different query should miss
        result = self.cache.get("How does quantum computing work?")
        assert result is None

    def test_ttl_expiration(self):
        # Use very short TTL
        cache = SemanticCache(embedder=self.embedder, ttl_seconds=0, similarity_threshold=0.9)
        cache.put("test query", {"answer": "test"})

        # Should be expired immediately
        time.sleep(0.01)
        result = cache.get("test query")
        assert result is None

    def test_multiple_entries(self):
        self.cache.put("What is chunking?", {"answer": "chunking answer"})
        self.cache.put("What is embedding?", {"answer": "embedding answer"})
        self.cache.put("What is reranking?", {"answer": "reranking answer"})

        # Each exact query should hit its own entry
        r1 = self.cache.get("What is chunking?")
        assert r1 is not None
        assert r1["answer"] == "chunking answer"

    def test_cosine_similarity_calculation(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert self.cache._cosine_similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0])
        assert self.cache._cosine_similarity(a, c) == pytest.approx(0.0)

    def test_graceful_fallback_without_redis(self):
        # Should not raise even with invalid redis URL
        cache = SemanticCache(
            embedder=self.embedder,
            redis_url="redis://nonexistent:6379",
        )
        assert cache.redis_client is None
        # Falls back to memory cache
        cache.put("test", {"answer": "hi"})
        result = cache.get("test")
        assert result is not None
