"""Document management API: upload, list, delete, and re-ingest documents.

Provides CRUD operations for the ``data/raw/`` document store and
triggers re-ingestion so the FAISS + BM25 indexes stay in sync.
"""

from __future__ import annotations

from pathlib import Path

from app.observability.logging import get_logger

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".docx"}

log = get_logger("rag.documents")


class DocumentManager:
    """Manages the raw document store on disk."""

    def __init__(self, raw_data_dir: str):
        self.raw_dir = Path(raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # --- List ---

    def list_documents(self) -> list[dict]:
        """Return metadata for every document in the raw store."""
        docs = []
        for f in sorted(self.raw_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                docs.append(
                    {
                        "filename": f.name,
                        "size_bytes": f.stat().st_size,
                        "extension": f.suffix.lower(),
                    }
                )
        return docs

    # --- Upload ---

    def save_document(self, filename: str, content: bytes) -> dict:
        """Save uploaded content to the raw store.

        Returns metadata dict. Raises ValueError for disallowed extensions.
        """
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        safe_name = Path(filename).name  # strip any path components
        dest = self.raw_dir / safe_name

        dest.write_bytes(content)
        log.info("document_saved", filename=safe_name, size=len(content))

        return {
            "filename": safe_name,
            "size_bytes": len(content),
            "extension": ext,
        }

    # --- Delete ---

    def delete_document(self, filename: str) -> bool:
        """Delete a document by name. Returns True if deleted, False if not found."""
        safe_name = Path(filename).name
        target = self.raw_dir / safe_name

        if not target.exists() or not target.is_file():
            return False

        # Ensure we only delete within raw_dir (path traversal guard)
        if not target.resolve().is_relative_to(self.raw_dir.resolve()):
            raise ValueError("Invalid filename")

        target.unlink()
        log.info("document_deleted", filename=safe_name)
        return True

    # --- Info ---

    def get_document_info(self, filename: str) -> dict | None:
        """Return metadata for a single document, or None if not found."""
        safe_name = Path(filename).name
        target = self.raw_dir / safe_name
        if not target.exists() or not target.is_file():
            return None
        return {
            "filename": target.name,
            "size_bytes": target.stat().st_size,
            "extension": target.suffix.lower(),
        }
