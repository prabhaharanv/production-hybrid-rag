from pathlib import Path
from unittest.mock import patch, MagicMock
from rag.loader import load_documents, _load_pdf, _load_html, _load_docx


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


class TestLoadHtml:
    def test_extracts_text_from_html(self, tmp_path):
        html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        (tmp_path / "page.html").write_text(html)
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert "Title" in docs[0]["text"]
        assert "Content here." in docs[0]["text"]
        assert docs[0]["metadata"]["file_type"] == ".html"

    def test_strips_script_and_style(self, tmp_path):
        html = "<html><head><style>body{}</style></head><body><script>alert(1)</script><p>Real content</p></body></html>"
        (tmp_path / "page.html").write_text(html)
        docs = load_documents(str(tmp_path))
        assert "alert" not in docs[0]["text"]
        assert "body{}" not in docs[0]["text"]
        assert "Real content" in docs[0]["text"]

    def test_strips_nav_footer_header(self, tmp_path):
        html = "<html><body><nav>Nav links</nav><main><p>Main content</p></main><footer>Footer</footer></body></html>"
        (tmp_path / "page.htm").write_text(html)
        docs = load_documents(str(tmp_path))
        assert "Nav links" not in docs[0]["text"]
        assert "Footer" not in docs[0]["text"]
        assert "Main content" in docs[0]["text"]

    def test_load_html_function_directly(self, tmp_path):
        html = "<p>Direct test</p>"
        p = tmp_path / "test.html"
        p.write_text(html)
        result = _load_html(p)
        assert result == "Direct test"


class TestLoadPdf:
    def test_loads_pdf_file(self, tmp_path):
        """Test PDF loading with a real PDF from data/raw if available, else mock."""
        sample = Path("/Users/prvelu/Desktop/lm/production-hybrid-rag/data/raw/llm_architecture.pdf")
        if sample.exists():
            text = _load_pdf(sample)
            assert len(text) > 0
            assert "transformer" in text.lower() or "attention" in text.lower() or "model" in text.lower()
        else:
            # Mock fitz for CI where sample may not exist
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Page 1 content"
            mock_doc = MagicMock()
            mock_doc.__iter__ = lambda self: iter([mock_page])
            with patch("rag.loader.fitz.open", return_value=mock_doc):
                result = _load_pdf(tmp_path / "fake.pdf")
                assert result == "Page 1 content"

    def test_pdf_metadata(self, tmp_path):
        """Test that PDF files get correct metadata."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Hello PDF"
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        with patch("rag.loader.fitz.open", return_value=mock_doc):
            docs = load_documents(str(tmp_path))
            assert len(docs) == 1
            assert docs[0]["metadata"]["file_type"] == ".pdf"
            assert docs[0]["title"] == "test.pdf"


class TestLoadDocx:
    def test_loads_docx_file(self, tmp_path):
        """Test DOCX loading with a real file if available, else mock."""
        sample = Path("/Users/prvelu/Desktop/lm/production-hybrid-rag/data/raw/prompt_engineering.docx")
        if sample.exists():
            text = _load_docx(sample)
            assert len(text) > 0
            assert "prompt" in text.lower() or "shot" in text.lower()
        else:
            mock_doc = MagicMock()
            mock_para1 = MagicMock()
            mock_para1.text = "First paragraph"
            mock_para2 = MagicMock()
            mock_para2.text = "Second paragraph"
            mock_doc.paragraphs = [mock_para1, mock_para2]
            with patch("rag.loader.DocxDocument", return_value=mock_doc):
                result = _load_docx(tmp_path / "fake.docx")
                assert "First paragraph" in result
                assert "Second paragraph" in result

    def test_docx_skips_empty_paragraphs(self):
        mock_doc = MagicMock()
        mock_para1 = MagicMock()
        mock_para1.text = "Content"
        mock_para2 = MagicMock()
        mock_para2.text = "   "
        mock_para3 = MagicMock()
        mock_para3.text = "More content"
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3]
        with patch("rag.loader.DocxDocument", return_value=mock_doc):
            result = _load_docx(Path("fake.docx"))
            assert "Content" in result
            assert "More content" in result
            # Empty paragraph should not appear
            lines = [line for line in result.split("\n\n") if line.strip()]
            assert len(lines) == 2
