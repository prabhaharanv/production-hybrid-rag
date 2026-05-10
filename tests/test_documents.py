"""Tests for document management."""

from rag.documents import DocumentManager


class TestDocumentManager:
    def setup_method(self, tmp_path=None):
        import tempfile

        self.tmp = tempfile.mkdtemp()
        self.manager = DocumentManager(self.tmp)

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_empty(self):
        assert self.manager.list_documents() == []

    def test_save_and_list(self):
        self.manager.save_document("test.txt", b"hello world")
        docs = self.manager.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.txt"
        assert docs[0]["size_bytes"] == 11
        assert docs[0]["extension"] == ".txt"

    def test_save_returns_metadata(self):
        result = self.manager.save_document("notes.md", b"# Title")
        assert result["filename"] == "notes.md"
        assert result["size_bytes"] == 7
        assert result["extension"] == ".md"

    def test_save_rejects_bad_extension(self):
        import pytest

        with pytest.raises(ValueError, match="Unsupported file type"):
            self.manager.save_document("evil.exe", b"binary")

    def test_save_strips_path_components(self):
        self.manager.save_document("../../../etc/passwd.txt", b"safe content")
        docs = self.manager.list_documents()
        assert len(docs) == 1
        assert docs[0]["filename"] == "passwd.txt"

    def test_delete_existing(self):
        self.manager.save_document("test.txt", b"hello")
        assert self.manager.delete_document("test.txt") is True
        assert self.manager.list_documents() == []

    def test_delete_nonexistent(self):
        assert self.manager.delete_document("nope.txt") is False

    def test_get_info_existing(self):
        self.manager.save_document("data.txt", b"content")
        info = self.manager.get_document_info("data.txt")
        assert info is not None
        assert info["filename"] == "data.txt"
        assert info["size_bytes"] == 7

    def test_get_info_nonexistent(self):
        assert self.manager.get_document_info("nope.txt") is None

    def test_allowed_extensions(self):
        for ext in [".txt", ".md", ".pdf", ".html", ".docx"]:
            self.manager.save_document(f"file{ext}", b"content")
        assert len(self.manager.list_documents()) == 5

    def test_overwrite_existing(self):
        self.manager.save_document("test.txt", b"v1")
        self.manager.save_document("test.txt", b"version2")
        docs = self.manager.list_documents()
        assert len(docs) == 1
        assert docs[0]["size_bytes"] == 8

    def test_list_sorted(self):
        self.manager.save_document("bravo.txt", b"b")
        self.manager.save_document("alpha.txt", b"a")
        docs = self.manager.list_documents()
        assert docs[0]["filename"] == "alpha.txt"
        assert docs[1]["filename"] == "bravo.txt"
