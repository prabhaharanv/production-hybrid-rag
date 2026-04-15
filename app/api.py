from fastapi import FastAPI, HTTPException
from app.config import settings
from app.schemas import AskRequest, AskResponse
from rag.embeddings import SentenceTransformerEmbedder
from rag.vector_store import FaissVectorStore
from rag.bm25_retriever import BM25Store
from rag.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from rag.generator import LLMGenerator
from rag.pipeline import RAGPipeline

app = FastAPI(title="Hybrid RAG API", version="0.2.0")

pipeline: RAGPipeline | None = None


@app.on_event("startup")
def startup_event():
    global pipeline

    try:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        vector_store = FaissVectorStore.load(settings.index_dir)
        bm25_store = BM25Store.load(settings.index_dir)

        dense = DenseRetriever(embedder=embedder, vector_store=vector_store)
        sparse = SparseRetriever(bm25_store=bm25_store)
        retriever = HybridRetriever(dense=dense, sparse=sparse, rrf_k=settings.rrf_k)

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        generator = LLMGenerator(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

        pipeline = RAGPipeline(retriever=retriever, generator=generator)
    except Exception as e:
        print(f"Startup failed: {e}")
        pipeline = None


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    top_k = request.top_k or settings.top_k
    result = pipeline.ask(request.question, top_k=top_k)
    return result