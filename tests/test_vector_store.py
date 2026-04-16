import pytest
import numpy as np
from rag.vector_store import FaissVectorStore


@pytest.fixture
def sample_store():
    store = FaissVectorStore(dim=4)
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32")
    records = [
        {"chunk_id": "c1", "doc_id": "d1", "title": "a", "source": "a.txt", "text": "first"},
        {"chunk_id": "c2", "doc_id": "d1", "title": "a", "source": "a.txt", "text": "second"},
        {"chunk_id": "c3", "doc_id": "d2", "title": "b", "source": "b.txt", "text": "third"},
    ]
    store.add(vectors, records)
    return store


class TestFaissVectorStore:
    def test_add_and_search(self, sample_store):
        query = np.array([[1, 0, 0, 0]], dtype="float32")
        results = sample_store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"

    def test_search_returns_scores(self, sample_store):
        query = np.array([[1, 0, 0, 0]], dtype="float32")
        results = sample_store.search(query, top_k=1)
        assert "score" in results[0]
        assert results[0]["score"] > 0

    def test_search_top_k(self, sample_store):
        query = np.array([[0.5, 0.5, 0, 0]], dtype="float32")
        results = sample_store.search(query, top_k=2)
        assert len(results) == 2

    def test_add_mismatch_raises(self):
        store = FaissVectorStore(dim=4)
        vectors = np.array([[1, 0, 0, 0]], dtype="float32")
        records = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        with pytest.raises(ValueError, match="mismatch"):
            store.add(vectors, records)

    def test_save_and_load(self, sample_store, tmp_path):
        sample_store.save(str(tmp_path / "index"))
        loaded = FaissVectorStore.load(str(tmp_path / "index"))
        assert len(loaded.records) == 3
        query = np.array([[1, 0, 0, 0]], dtype="float32")
        results = loaded.search(query, top_k=1)
        assert results[0]["chunk_id"] == "c1"
