from rag.loader import load_documents
from rag.chunking import chunk_documents
from rag.embeddings import SentenceTransformerEmbedder
from rag.vector_store import FaissVectorStore
from rag.bm25_retriever import BM25Store


def run_ingestion(
    raw_data_dir: str,
    index_dir: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    documents = load_documents(raw_data_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=chunk_overlap)

    if not chunks:
        raise ValueError("No chunks created. Check your input data.")

    embedder = SentenceTransformerEmbedder(embedding_model)
    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.embed_documents(texts)

    vector_store = FaissVectorStore(dim=vectors.shape[1])
    vector_store.add(vectors, chunks)
    vector_store.save(index_dir)

    bm25_store = BM25Store(records=chunks)
    bm25_store.save(index_dir)

    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
        "index_dir": index_dir,
    }