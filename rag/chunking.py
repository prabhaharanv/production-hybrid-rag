def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 400, overlap: int = 60) -> list[dict]:
    chunked = []

    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            chunked.append(
                {
                    "chunk_id": f'{doc["doc_id"]}_chunk_{idx}',
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "text": chunk,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": idx,
                    },
                }
            )

    return chunked