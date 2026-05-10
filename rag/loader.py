from pathlib import Path

import fitz  # pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm", ".docx"}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _load_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages).strip()


def _load_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    # Remove script/style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _load_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_text(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _load_text(path)
    elif ext == ".pdf":
        return _load_pdf(path)
    elif ext in (".html", ".htm"):
        return _load_html(path)
    elif ext == ".docx":
        return _load_docx(path)
    return None


def load_documents(data_dir: str) -> list[dict]:
    documents = []

    for path in Path(data_dir).rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = _extract_text(path)
        if not text:
            continue

        documents.append(
            {
                "doc_id": path.stem,
                "title": path.name,
                "source": str(path),
                "text": text,
                "metadata": {
                    "file_type": path.suffix.lower(),
                    "parent_dir": str(path.parent),
                },
            }
        )

    return documents
