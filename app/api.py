from contextlib import asynccontextmanager
import os

import structlog
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse, Response

from app.config import settings
from app.schemas import AskRequest, AskResponse
from app.observability.tracing import init_tracing, trace_span
from app.observability.metrics import init_metrics, get_metrics, track_request
from app.observability.logging import init_logging, get_logger, new_correlation_id
from app.observability.health import HealthChecker
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

# ---- Health checker ----
health_checker = HealthChecker()


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

    # Initialise observability
    log_json = os.getenv("LOG_JSON", "true").lower() == "true"
    init_logging(log_level=os.getenv("LOG_LEVEL", "INFO"), json_output=log_json)
    init_tracing(
        service_name="rag-api",
        otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
    )
    metrics = init_metrics()
    log = get_logger("app.api")

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
        health_checker.set_pipeline(pipeline)
        metrics.pipeline_ready.set(1)
        log.info("pipeline_started")
    except Exception as e:
        log.error("startup_failed", error=str(e))
        pipeline = None
        health_checker.set_pipeline(None)
        metrics.pipeline_ready.set(0)

    yield

    metrics.pipeline_ready.set(0)
    health_checker.set_pipeline(None)
    pipeline = None
    log.info("pipeline_stopped")


# ---- App ----
app = FastAPI(title="Hybrid RAG API", version="0.5.0", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})


# ---- Observability endpoints ----

@app.get("/health")
def health():
    return health_checker.liveness()


@app.get("/health/ready")
def readiness():
    result = health_checker.readiness()
    status_code = 200 if result["healthy"] else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/metrics")
def metrics_endpoint():
    m = get_metrics()
    return Response(content=m.generate_latest(), media_type=m.content_type)


# ---- Main endpoint ----

@app.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit)
def ask(request: AskRequest, req: Request, _api_key: str | None = Depends(verify_api_key)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    correlation_id = new_correlation_id()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    log = get_logger("app.api")

    top_k = request.top_k or settings.top_k
    log.info("request_received", question_len=len(request.question), top_k=top_k)

    with track_request() as m:
        with trace_span("ask", {"correlation_id": correlation_id, "top_k": top_k}):
            result = pipeline.ask(request.question, top_k=top_k)

    return result