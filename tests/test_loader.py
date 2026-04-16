import pytest
import tempfile
from pathlib import Path
from rag.loader import load_documents


class TestLoadDocuments:
    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "test.txt").write_text("hello world")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["text"] == "hello world"
        assert docs[0]["title"] == "test.txt"

    def test_loads_md_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Title")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["text"] == "# Title"

    def test_ignores_other_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "test.txt").write_text("hello")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_ignores_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 0

    def test_recursive(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested content")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1

    def test_doc_fields(self, tmp_path):
        (tmp_path / "doc.txt").write_text("content")
        docs = load_documents(str(tmp_path))
        doc = docs[0]
        assert "doc_id" in doc
        assert "title" in doc
        assert "source" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert doc["metadata"]["file_type"] == ".txt"
