from rag.retriever import (
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)


# ---- RRF tests ----

class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [
            {"chunk_id": "a", "text": "A", "score": 0.9},
            {"chunk_id": "b", "text": "B", "score": 0.5},
        ]
        fused = reciprocal_rank_fusion([results], k=60)
        assert [r["chunk_id"] for r in fused] == ["a", "b"]

    def test_two_lists_same_items(self):
        list1 = [
            {"chunk_id": "a", "text": "A", "score": 0.9},
            {"chunk_id": "b", "text": "B", "score": 0.5},
        ]
        list2 = [
            {"chunk_id": "b", "text": "B", "score": 0.8},
            {"chunk_id": "a", "text": "A", "score": 0.3},
        ]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [r["chunk_id"] for r in fused]
        assert set(ids) == {"a", "b"}

    def test_deduplication(self):
        list1 = [{"chunk_id": "a", "text": "A", "score": 0.9}]
        list2 = [{"chunk_id": "a", "text": "A", "score": 0.8}]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 1
        assert fused[0]["chunk_id"] == "a"

    def test_scores_are_positive(self):
        results = [{"chunk_id": "a", "text": "A", "score": 0.5}]
        fused = reciprocal_rank_fusion([results], k=60)
        assert all(r["score"] > 0 for r in fused)

    def test_fused_score_higher_when_in_multiple_lists(self):
        list1 = [
            {"chunk_id": "a", "text": "A", "score": 0.9},
            {"chunk_id": "b", "text": "B", "score": 0.5},
        ]
        list2 = [
            {"chunk_id": "a", "text": "A", "score": 0.7},
        ]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        scores = {r["chunk_id"]: r["score"] for r in fused}
        # "a" appears in both lists, "b" only in one → "a" should score higher
        assert scores["a"] > scores["b"]

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_original_score_stripped(self):
        results = [{"chunk_id": "a", "text": "A", "score": 0.99}]
        fused = reciprocal_rank_fusion([results], k=60)
        # The fused score should be the RRF score, not the original
        assert fused[0]["score"] != 0.99


# ---- Retriever tests with stubs ----

class FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self, results):
        self._results = results

    def search(self, query_vector, top_k=5):
        return self._results[:top_k]


class FakeBM25Store:
    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=5):
        return self._results[:top_k]


class TestDenseRetriever:
    def test_retrieve_returns_results(self):
        results = [{"chunk_id": "c1", "text": "hello", "score": 0.8}]
        dense = DenseRetriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(results))
        out = dense.retrieve("query", top_k=1)
        assert len(out) == 1
        assert out[0]["chunk_id"] == "c1"


class TestSparseRetriever:
    def test_retrieve_returns_results(self):
        results = [{"chunk_id": "c2", "text": "world", "score": 0.6}]
        sparse = SparseRetriever(bm25_store=FakeBM25Store(results))
        out = sparse.retrieve("query", top_k=1)
        assert len(out) == 1
        assert out[0]["chunk_id"] == "c2"


class TestHybridRetriever:
    def test_combines_dense_and_sparse(self):
        dense_results = [
            {"chunk_id": "c1", "text": "A", "score": 0.9},
            {"chunk_id": "c2", "text": "B", "score": 0.5},
        ]
        sparse_results = [
            {"chunk_id": "c2", "text": "B", "score": 0.8},
            {"chunk_id": "c3", "text": "C", "score": 0.3},
        ]
        dense = DenseRetriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(dense_results))
        sparse = SparseRetriever(bm25_store=FakeBM25Store(sparse_results))
        hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60)

        out = hybrid.retrieve("query", top_k=3)
        ids = [r["chunk_id"] for r in out]
        assert "c2" in ids  # present in both lists, should rank high

    def test_top_k_limits_results(self):
        dense_results = [{"chunk_id": f"d{i}", "text": f"D{i}", "score": 0.9 - i * 0.1} for i in range(5)]
        sparse_results = [{"chunk_id": f"s{i}", "text": f"S{i}", "score": 0.9 - i * 0.1} for i in range(5)]
        dense = DenseRetriever(embedder=FakeEmbedder(), vector_store=FakeVectorStore(dense_results))
        sparse = SparseRetriever(bm25_store=FakeBM25Store(sparse_results))
        hybrid = HybridRetriever(dense=dense, sparse=sparse, rrf_k=60)

        out = hybrid.retrieve("query", top_k=2)
        assert len(out) == 2
