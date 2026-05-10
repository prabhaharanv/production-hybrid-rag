"""Semantic caching with Redis and embedding similarity.

If a semantically similar question was previously asked, returns the cached
answer — saving LLM cost and latency. Uses cosine similarity between query
embeddings to detect cache hits.

Architecture:
- Queries are embedded and stored in Redis alongside their answers
- On each new query, its embedding is compared to cached embeddings
- If similarity exceeds a threshold, the cached answer is returned
- Cache entries have a configurable TTL

Falls back gracefully if Redis is unavailable (cache miss = normal pipeline).
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np


class SemanticCache:
    """In-memory semantic cache with optional Redis backend.

    Uses embedding similarity to detect semantically equivalent queries.
    Falls back to in-memory store if Redis is not available.
    """

    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        redis_url: str | None = None,
    ):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.redis_client = None

        if redis_url:
            try:
                import redis

                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None

        # In-memory fallback cache
        self._memory_cache: list[dict] = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        dot = np.dot(a_flat, b_flat)
        norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def get(self, query: str) -> dict | None:
        """Look up a semantically similar cached result."""
        query_embedding = self.embedder.embed_query(query)

        if self.redis_client:
            return self._redis_get(query_embedding)
        return self._memory_get(query_embedding)

    def put(self, query: str, result: dict) -> None:
        """Cache a query-result pair."""
        query_embedding = self.embedder.embed_query(query)

        if self.redis_client:
            self._redis_put(query, query_embedding, result)
        else:
            self._memory_put(query, query_embedding, result)

    def _memory_get(self, query_embedding: np.ndarray) -> dict | None:
        now = time.time()
        # Clean expired entries
        self._memory_cache = [
            e for e in self._memory_cache if now - e["timestamp"] < self.ttl_seconds
        ]

        best_score = 0.0
        best_entry = None

        for entry in self._memory_cache:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            return {
                **best_entry["result"],
                "_cache_hit": True,
                "_cache_score": best_score,
            }
        return None

    def _memory_put(self, query: str, embedding: np.ndarray, result: dict) -> None:
        self._memory_cache.append(
            {
                "query": query,
                "embedding": embedding,
                "result": result,
                "timestamp": time.time(),
            }
        )

    def _redis_get(self, query_embedding: np.ndarray) -> dict | None:
        try:
            keys = self.redis_client.keys("semantic_cache:*")
            best_score = 0.0
            best_result = None

            for key in keys:
                raw = self.redis_client.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                cached_emb = np.array(entry["embedding"], dtype="float32")
                score = self._cosine_similarity(query_embedding, cached_emb)
                if score > best_score:
                    best_score = score
                    best_result = entry["result"]

            if best_result and best_score >= self.similarity_threshold:
                return {**best_result, "_cache_hit": True, "_cache_score": best_score}
        except Exception:
            pass
        return None

    def _redis_put(self, query: str, embedding: np.ndarray, result: dict) -> None:
        try:
            key = f"semantic_cache:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
            entry = {
                "query": query,
                "embedding": embedding.tolist(),
                "result": result,
            }
            self.redis_client.setex(key, self.ttl_seconds, json.dumps(entry))
        except Exception:
            # Fall back to memory if Redis write fails
            self._memory_put(query, embedding, result)

    def invalidate(self) -> None:
        """Clear all cached entries."""
        self._memory_cache.clear()
        if self.redis_client:
            try:
                keys = self.redis_client.keys("semantic_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                pass

    @property
    def size(self) -> int:
        if self.redis_client:
            try:
                return len(self.redis_client.keys("semantic_cache:*"))
            except Exception:
                pass
        return len(self._memory_cache)
