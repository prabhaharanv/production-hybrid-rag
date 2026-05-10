from contextlib import asynccontextmanager
import os

import structlog
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Security,
    Depends,
    Body,
    UploadFile,
    File,
)
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse, Response, StreamingResponse

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
from rag.documents import DocumentManager
from rag.memory import ConversationMemory

# ---- Rate limiter ----
limiter = Limiter(key_func=get_remote_address)

# ---- API Key auth ----
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---- Health checker ----
health_checker = HealthChecker()

# ---- Document manager ----
doc_manager = DocumentManager(settings.raw_data_dir)

# ---- Conversation memory ----
memory = ConversationMemory(
    max_turns=settings.memory_max_turns,
    ttl=settings.memory_ttl,
)


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

        reranker = (
            Reranker(settings.reranker_model) if settings.enable_reranker else None
        )

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
app = FastAPI(title="Hybrid RAG API", version="0.7.0", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429, content={"detail": "Rate limit exceeded. Try again later."}
    )


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
def ask(
    request: Request,
    body: AskRequest = Body(...),
    _api_key: str | None = Depends(verify_api_key),
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    correlation_id = new_correlation_id()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    log = get_logger("app.api")

    top_k = body.top_k or settings.top_k
    log.info("request_received", question_len=len(body.question), top_k=top_k)

    with track_request():
        with trace_span("ask", {"correlation_id": correlation_id, "top_k": top_k}):
            result = pipeline.ask(
                body.question,
                top_k=top_k,
                conversation_id=body.conversation_id,
                memory=memory if settings.enable_conversation_memory else None,
            )

    return result


# ---- Streaming endpoint ----


@app.post("/ask/stream")
@limiter.limit(settings.rate_limit)
def ask_stream(
    request: Request,
    body: AskRequest = Body(...),
    _api_key: str | None = Depends(verify_api_key),
):
    """Stream the RAG response using Server-Sent Events (SSE)."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    correlation_id = new_correlation_id()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    log = get_logger("app.api")

    top_k = body.top_k or settings.top_k
    log.info("stream_request_received", question_len=len(body.question), top_k=top_k)

    def event_generator():
        with track_request():
            with trace_span(
                "ask_stream", {"correlation_id": correlation_id, "top_k": top_k}
            ):
                yield from pipeline.ask_stream(body.question, top_k=top_k)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---- Document management endpoints ----


@app.get("/documents")
def list_documents(_api_key: str | None = Depends(verify_api_key)):
    """List all documents in the raw store."""
    return doc_manager.list_documents()


@app.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    _api_key: str | None = Depends(verify_api_key),
):
    """Upload a document to the raw store."""
    content = await file.read()
    try:
        result = doc_manager.save_document(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/documents/{filename}")
def get_document_info(filename: str, _api_key: str | None = Depends(verify_api_key)):
    """Get metadata for a specific document."""
    info = doc_manager.get_document_info(filename)
    if info is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return info


@app.delete("/documents/{filename}")
def delete_document(filename: str, _api_key: str | None = Depends(verify_api_key)):
    """Delete a document from the raw store."""
    deleted = doc_manager.delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": filename}


# ---- Conversation memory endpoints ----


@app.get("/conversations")
def list_conversations(_api_key: str | None = Depends(verify_api_key)):
    """List active conversations."""
    return memory.list_conversations()


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str, _api_key: str | None = Depends(verify_api_key)
):
    """Get the history for a conversation."""
    turns = memory.get_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "turns": [
            {"question": t.question, "answer": t.answer, "timestamp": t.timestamp}
            for t in turns
        ],
    }


@app.delete("/conversations/{conversation_id}")
def clear_conversation(
    conversation_id: str, _api_key: str | None = Depends(verify_api_key)
):
    """Clear a conversation's history."""
    memory.clear(conversation_id)
    return {"cleared": conversation_id}
