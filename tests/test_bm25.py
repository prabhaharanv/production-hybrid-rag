import pytest
from rag.bm25_retriever import BM25Store, _tokenize


class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []


class TestBM25Store:
    @pytest.fixture
    def sample_store(self):
        records = [
            {"chunk_id": "c1", "text": "the cat sat on the mat"},
            {"chunk_id": "c2", "text": "the dog chased the cat"},
            {"chunk_id": "c3", "text": "the bird flew away"},
        ]
        return BM25Store(records=records)

    def test_search_returns_relevant(self, sample_store):
        results = sample_store.search("cat", top_k=2)
        ids = [r["chunk_id"] for r in results]
        assert "c1" in ids
        assert "c2" in ids

    def test_search_scores_descending(self, sample_store):
        results = sample_store.search("cat sat", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_zero_score_excluded(self, sample_store):
        results = sample_store.search("zebra", top_k=3)
        assert len(results) == 0

    def test_empty_store_returns_empty(self):
        store = BM25Store()
        assert store.search("anything") == []

    def test_add_rebuilds_index(self):
        store = BM25Store(records=[
            {"chunk_id": "c1", "text": "the cat sat on the mat"},
            {"chunk_id": "c2", "text": "the dog chased the ball"},
        ])
        store.add([{"chunk_id": "c3", "text": "the moon shines at night"}])
        results = store.search("moon", top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c3"

    def test_save_and_load(self, sample_store, tmp_path):
        sample_store.save(str(tmp_path / "index"))
        loaded = BM25Store.load(str(tmp_path / "index"))
        assert len(loaded.records) == 3
        results = loaded.search("cat", top_k=1)
        assert results[0]["chunk_id"] in ["c1", "c2"]
