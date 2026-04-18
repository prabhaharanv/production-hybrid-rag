from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from app.config import settings
from app.schemas import AskRequest, AskResponse
from rag.embeddings import SentenceTransformerEmbedder
from rag.vector_store import FaissVectorStore
from rag.bm25_retriever import BM25Store
from rag.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from rag.reranker import Reranker
from rag.query_rewriter import QueryRewriter
from rag.generator import LLMGenerator
from rag.pipeline import RAGPipeline

# ---- Rate limiter ----
limiter = Limiter(key_func=get_remote_address)

# ---- API Key auth ----
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Validate API key if RAG_API_KEY is configured. Skip auth if not set."""
    expected = settings.rag_api_key
    if not expected:
        return None  # auth disabled
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---- Lifespan ----
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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

        reranker = Reranker(settings.reranker_model) if settings.enable_reranker else None

        query_rewriter = (
            QueryRewriter(
                model=settings.llm_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            if settings.enable_query_rewriting
            else None
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            query_rewriter=query_rewriter,
        )
    except Exception as e:
        print(f"Startup failed: {e}")
        pipeline = None

    yield

    pipeline = None


# ---- App ----
app = FastAPI(title="Hybrid RAG API", version="0.4.0", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit)
def ask(request: AskRequest, req: Request, _api_key: str | None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    top_k = request.top_k or settings.top_k
    result = pipeline.ask(request.question, top_k=top_k)
    return result