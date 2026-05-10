"""Tests for parent-child chunking strategy."""

import tempfile

from rag.parent_child import ParentChildChunker, ParentChildStore


class TestParentChildChunker:
    def setup_method(self):
        self.chunker = ParentChildChunker(
            parent_chunk_size=20,  # Small for testing (words)
            parent_overlap=5,
            child_chunk_size=5,
            child_overlap=1,
        )
        self.docs = [
            {
                "doc_id": "doc1",
                "title": "test.txt",
                "source": "data/raw/test.txt",
                "text": " ".join(f"word{i}" for i in range(40)),
                "metadata": {"format": "txt"},
            }
        ]

    def test_chunk_documents_returns_children_and_mapping(self):
        children, mapping = self.chunker.chunk_documents(self.docs)

        assert len(children) > 0
        assert len(mapping) > 0
        # Every child should have a parent mapping
        for child in children:
            assert child["chunk_id"] in mapping

    def test_child_chunks_have_required_fields(self):
        children, _ = self.chunker.chunk_documents(self.docs)

        for child in children:
            assert "chunk_id" in child
            assert "doc_id" in child
            assert "title" in child
            assert "source" in child
            assert "text" in child
            assert "metadata" in child
            assert child["metadata"]["chunking_strategy"] == "parent_child"

    def test_child_text_is_subset_of_parent(self):
        children, mapping = self.chunker.chunk_documents(self.docs)

        for child in children:
            parent_text = mapping[child["chunk_id"]]
            # Child words should all appear in parent
            child_words = set(child["text"].split())
            parent_words = set(parent_text.split())
            assert child_words.issubset(parent_words)

    def test_preserves_document_metadata(self):
        children, _ = self.chunker.chunk_documents(self.docs)
        for child in children:
            assert child["doc_id"] == "doc1"
            assert child["title"] == "test.txt"


class TestParentChildStore:
    def test_expand_to_parents(self):
        mapping = {
            "child_1": "This is the full parent text.",
            "child_2": "Another parent text.",
        }
        store = ParentChildStore(mapping)

        chunks = [
            {
                "chunk_id": "child_1",
                "text": "full parent",
                "metadata": {"parent_id": "p1"},
            },
            {"chunk_id": "child_2", "text": "Another", "metadata": {"parent_id": "p2"}},
        ]

        expanded = store.expand_to_parents(chunks)
        assert expanded[0]["text"] == "This is the full parent text."
        assert expanded[1]["text"] == "Another parent text."

    def test_expand_deduplicates_same_parent(self):
        mapping = {
            "child_1": "Parent text.",
            "child_2": "Parent text.",
        }
        store = ParentChildStore(mapping)

        chunks = [
            {
                "chunk_id": "child_1",
                "text": "chunk1",
                "metadata": {"parent_id": "same_parent"},
            },
            {
                "chunk_id": "child_2",
                "text": "chunk2",
                "metadata": {"parent_id": "same_parent"},
            },
        ]

        expanded = store.expand_to_parents(chunks)
        # Only one result since both children map to same parent
        assert len(expanded) == 1
        assert expanded[0]["text"] == "Parent text."

    def test_passthrough_if_no_parent_mapping(self):
        store = ParentChildStore({})
        chunks = [{"chunk_id": "unknown", "text": "original", "metadata": {}}]
        expanded = store.expand_to_parents(chunks)
        assert expanded[0]["text"] == "original"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping = {"c1": "parent 1", "c2": "parent 2"}
            store = ParentChildStore(mapping)
            store.save(tmpdir)

            loaded = ParentChildStore.load(tmpdir)
            assert loaded.child_to_parent == mapping

    def test_load_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParentChildStore.load(tmpdir)
            assert store.child_to_parent == {}
