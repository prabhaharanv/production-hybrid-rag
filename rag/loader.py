from pathlib import Path


def load_documents(data_dir: str) -> list[dict]:
    documents = []

    for path in Path(data_dir).rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".txt", ".md"]:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore").strip()
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