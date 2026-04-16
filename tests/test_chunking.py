import pytest
from rag.chunking import chunk_text, chunk_documents


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        result = chunk_text("hello world", chunk_size=10)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_exact_chunk_size(self):
        text = " ".join(["word"] * 10)
        result = chunk_text(text, chunk_size=10, overlap=0)
        assert len(result) == 1

    def test_overlap_produces_more_chunks(self):
        text = " ".join(["word"] * 20)
        no_overlap = chunk_text(text, chunk_size=10, overlap=0)
        with_overlap = chunk_text(text, chunk_size=10, overlap=5)
        assert len(with_overlap) > len(no_overlap)

    def test_chunks_contain_all_words(self):
        words = [f"w{i}" for i in range(15)]
        text = " ".join(words)
        result = chunk_text(text, chunk_size=10, overlap=3)
        all_text = " ".join(result)
        for w in words:
            assert w in all_text


class TestChunkDocuments:
    def test_single_document(self):
        docs = [{"doc_id": "d1", "title": "test", "source": "test.txt", "text": "hello world"}]
        result = chunk_documents(docs, chunk_size=100)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "d1_chunk_0"
        assert result[0]["doc_id"] == "d1"

    def test_chunk_ids_are_unique(self):
        docs = [
            {"doc_id": "d1", "title": "a", "source": "a.txt", "text": " ".join(["word"] * 30)},
            {"doc_id": "d2", "title": "b", "source": "b.txt", "text": " ".join(["word"] * 30)},
        ]
        result = chunk_documents(docs, chunk_size=10, overlap=2)
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_metadata_has_chunk_index(self):
        docs = [{"doc_id": "d1", "title": "t", "source": "s", "text": " ".join(["word"] * 30)}]
        result = chunk_documents(docs, chunk_size=10, overlap=2)
        for i, chunk in enumerate(result):
            assert chunk["metadata"]["chunk_index"] == i
