# Production Hybrid RAG

A production-grade Retrieval-Augmented Generation system with hybrid search, cross-encoder reranking, query rewriting, source citations, and answer abstention.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.5.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-108%20passed-brightgreen)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram.

```
Question → API Key Auth → Rate Limiter → Query Rewriting → Hybrid Retrieval (FAISS + BM25 → RRF) → Cross-Encoder Reranking → LLM Generation → Abstention Check → Citation Extraction → Response
```

## Features

- **Ingestion**: Document loading (`.txt`, `.md`), word-level chunking with overlap
- **Embeddings**: Sentence-transformer (`all-MiniLM-L6-v2`) with FAISS indexing
- **Sparse retrieval**: BM25 keyword search (`rank-bm25`)
- **Hybrid retrieval**: Reciprocal Rank Fusion (RRF) merging dense + sparse results
- **Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) for precision
- **Query rewriting**: LLM rewrites vague queries before retrieval
- **Source citations**: Bracket notation `[1]`, `[2]` with structured citation objects
- **Answer abstention**: Refuses to answer when context is insufficient
- **LLM generation**: OpenAI or Ollama (local)
- **API**: FastAPI with Pydantic validation, lifespan-managed startup
- **API security**: API key authentication (`X-API-Key` header) and rate limiting (slowapi)
- **Docker**: Multi-stage build, non-root user, pinned dependencies, CVE-scanned (0 critical/high)
- **Observability**: OpenTelemetry tracing, Prometheus metrics, structlog JSON logging with correlation IDs
- **Monitoring**: Grafana dashboards (latency percentiles, error rates, abstention rate), Prometheus alerting rules
- **Health checks**: Liveness (`/health`) and deep readiness probes (`/health/ready`) for Kubernetes/load balancers
- **Evaluation**: Benchmark suite with keyword recall, source hit rate, and abstention accuracy
- **Deep evaluation**: Mathematical eval framework — RAGAS (faithfulness, answer relevance, context precision/recall), BERTScore, MRR, NDCG@K, NLI-based hallucination detection
- **Testing**: 83 unit tests across 9 test files (pytest)

## Project Structure

```
production-hybrid-rag/
├── app/
│   ├── api.py              # FastAPI application and endpoints
│   ├── config.py           # Settings loaded from .env
│   ├── schemas.py          # Request/response Pydantic models
│   └── observability/      # Observability & monitoring
│       ├── tracing.py      # OpenTelemetry distributed tracing
│       ├── metrics.py      # Prometheus metrics and counters
│       ├── logging.py      # structlog JSON logging with correlation IDs
│       └── health.py       # Liveness and readiness health checks
├── rag/
│   ├── loader.py           # Document loader (.txt, .md)
│   ├── chunking.py         # Text chunking with overlap
│   ├── embeddings.py       # Sentence-transformer wrapper
│   ├── vector_store.py     # FAISS index save/load/search
│   ├── bm25_retriever.py   # BM25 sparse index and search
│   ├── retriever.py        # Dense, Sparse, Hybrid retrievers + RRF
│   ├── reranker.py         # Cross-encoder reranker
│   ├── query_rewriter.py   # LLM-based query rewriting
│   ├── prompting.py        # Prompt with citation + abstention rules
│   ├── generator.py        # LLM generation (OpenAI-compatible)
│   ├── pipeline.py         # Rewrite → Retrieve → Rerank → Generate
│   └── ingest.py           # End-to-end ingestion orchestration
├── eval/
│   ├── dataset.json        # Evaluation questions with expected answers + ground truth
│   ├── metrics.py          # Mathematical evaluation metrics
│   └── benchmark.py        # Benchmark runner with metrics
├── tests/                  # Unit tests (pytest)
├── scripts/
│   └── ingest_docs.py      # CLI ingestion entry point
├── docs/
│   └── architecture.md     # Mermaid architecture diagram
├── data/
│   └── raw/                # Place source documents here
├── Dockerfile              # Multi-stage build, non-root user
├── docker-compose.yml      # App + Prometheus + Grafana + Jaeger + OTel Collector
├── .dockerignore
├── .env.example
├── monitoring/
│   ├── prometheus.yml      # Prometheus scrape config
│   ├── alerts.yml          # Alerting rules (latency, errors, abstention)
│   ├── otel-collector.yml  # OpenTelemetry Collector config
│   ├── grafana-dashboard.json
│   ├── grafana-datasources.yml
│   └── grafana-dashboards.yml
├── requirements.txt        # Pinned production dependencies
└── requirements-dev.txt    # Dev dependencies (pytest)
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt   # includes production + dev (pytest)

# 2. Configure environment
cp .env.example .env
# Edit .env with your API key (OpenAI or Ollama)

# 3. Add documents to data/raw/
#    Supports .txt and .md files

# 4. Run ingestion
python scripts/ingest_docs.py

# 5. Start API server
uvicorn app.api:app --reload
```

### Using Ollama (free, local)

```bash
brew install ollama
ollama serve          # in a separate terminal
ollama pull llama3.2:3b
```

Set in `.env`:
```
OPENAI_API_KEY=ollama
LLM_MODEL=llama3.2:3b
OPENAI_BASE_URL=http://localhost:11434/v1

# Optional: API security
RAG_API_KEY=your-secret-key   # omit to disable auth
RATE_LIMIT=20/minute           # requests per window
```

## API

### `POST /ask`

```bash
# Without auth (when RAG_API_KEY is not set)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 3}'

# With auth (when RAG_API_KEY is set)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"question": "What is RAG?", "top_k": 3}'
```

Response:
```json
{
  "question": "What is RAG?",
  "rewritten_query": "What is Retrieval-Augmented Generation RAG and how does it work?",
  "answer": "...",
  "abstained": false,
  "citations": [
    {
      "reference": 1,
      "title": "rag_intro.txt",
      "source": "data/raw/rag_intro.txt"
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "...",
      "doc_id": "...",
      "title": "...",
      "source": "...",
      "text": "...",
      "score": 0.85,
      "metadata": {}
    }
  ]
}
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

## Docker

```bash
# Start the full stack (app + Prometheus + Grafana + Jaeger + OTel Collector)
docker compose up --build
```

**First run**: The embedding model (`all-MiniLM-L6-v2`) and reranker model (`ms-marco-MiniLM-L-6-v2`) are downloaded automatically from HuggingFace on first startup (~100 MB total). Subsequent starts are instant because models are cached on the host via the `~/.cache/huggingface` volume mount.

**Using Ollama (local LLM)**: Ollama runs on your host machine, not inside Docker. Start it before the stack:

```bash
ollama serve                        # terminal 1
ollama pull llama3.2:3b             # one-time download
docker compose up --build           # terminal 2
```

Set `OPENAI_BASE_URL=http://host.docker.internal:11434/v1` in `.env` so the container can reach Ollama on the host.

**Standalone** (without the monitoring stack):
```bash
docker build -t hybrid-rag .
docker run -p 8000:8000 --env-file .env \
  -v ./data:/app/data \
  -v ~/.cache/huggingface:/home/appuser/.cache/huggingface \
  hybrid-rag
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_pipeline.py -v
```

## Observability & Monitoring

The full monitoring stack runs alongside the app via Docker Compose:

```bash
docker compose up -d
```

| Service | URL | Purpose |
|---------|-----|---------|
| Prometheus | http://localhost:9090 | Metrics scraping and alerting |
| Grafana | http://localhost:3000 (admin/admin) | Dashboards: latency percentiles, error rates, abstention rate, token usage |
| Jaeger | http://localhost:16686 | Trace explorer: see every pipeline step with timing |

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe — is the process alive? |
| `GET /health/ready` | Readiness probe — are pipeline, retriever, generator, and indexes all operational? Returns 503 if degraded |
| `GET /metrics` | Prometheus scrape endpoint |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTLP_ENDPOINT` | *(none)* | OTel Collector gRPC endpoint (e.g. `http://otel-collector:4317`). Console output if unset |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_JSON` | `true` | JSON structured logs (`false` for coloured console) |

### Alerting Rules

Pre-configured in `monitoring/alerts.yml`:

- **RAGHighLatencyP95/P99**: p95 > 5s or p99 > 10s for 5 minutes
- **RAGHighErrorRate**: Error rate > 5% for 5 minutes
- **RAGHighAbstentionRate**: Abstention rate > 30% for 10 minutes
- **RAGPipelineNotReady**: Pipeline readiness gauge at 0 for 2 minutes
- **RAGSlowRetrieval / RAGSlowGeneration**: Individual step latency spikes

## Evaluation

```bash
# Run the benchmark suite (requires ingested data + LLM access)
python eval/benchmark.py

# Use a custom dataset
python eval/benchmark.py path/to/dataset.json

# Enable deep evaluation metrics
python eval/benchmark.py --deep-eval

# Combine options
python eval/benchmark.py eval/dataset.json --deep-eval --top-k 3
```

### Basic Metrics

- **Keyword recall**: Fraction of expected keywords found in the answer
- **Source hit rate**: Whether the correct source document was cited
- **Abstention accuracy**: Whether the system correctly refused out-of-scope questions
- **Latency**: End-to-end time per question

### Mathematical Evaluation Framework

Enable with `--deep-eval`. These metrics provide rigorous, quantitative evaluation of RAG quality.

| Metric | Formula | What it proves |
|--------|---------|----------------|
| **RAGAS Faithfulness** | $\text{Faithfulness} = \frac{\text{claims supported by context}}{\text{total claims}}$ | Answer is grounded in retrieved context |
| **RAGAS Answer Relevance** | $\text{Relevance} = \frac{1}{N} \sum_{i=1}^{N} \cos(E(q), E(a_i))$ where $a_i$ are reverse-generated questions | Answer actually addresses the question |
| **Context Precision** | $\text{CP@K} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}(k) \cdot \text{rel}(k)$ | Retrieved chunks are relevant |
| **Context Recall** | $\text{CR} = \frac{\text{ground truth sentences attributable to context}}{\text{total ground truth sentences}}$ | Context covers the ground truth |
| **BERTScore** | $F_{\text{BERT}} = \frac{2 \cdot P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$ using contextual embeddings | Semantic similarity beyond keywords |
| **MRR & NDCG@K** | $\text{MRR} = \frac{1}{\lvert Q \rvert} \sum_{i=1}^{\lvert Q \rvert} \frac{1}{\text{rank}_i}$ , $\text{NDCG@K} = \frac{DCG@K}{IDCG@K}$ | Retrieval ranking quality |
| **Hallucination Detection** | NLI-based: classify each claim as entailed / contradicted / neutral vs context | Catches fabricated facts |

#### Implementation Details

- **NLI backbone**: `cross-encoder/nli-deberta-v3-small` classifies (context, claim) pairs
- **Embedding model**: `all-MiniLM-L6-v2` for answer relevance cosine similarity and BERTScore
- **Ground truth**: `eval/dataset.json` includes `ground_truth` fields for context recall
- Models are lazily loaded (first use only) and reused across all evaluation items
- Abstained answers are excluded from deep eval (no generated content to evaluate)

#### Example Output

```
=== Benchmark Summary ===
  total_questions: 10
  avg_keyword_recall: 0.875
  source_hit_rate: 0.9
  abstention_accuracy: 1.0
  avg_latency_s: 1.23
  total_latency_s: 12.3
  --- Deep Eval Averages ---
    faithfulness: 0.8750
    answer_relevance: 0.7321
    context_precision: 0.9125
    context_recall: 0.8500
    bertscore_f1: 0.8234
    mrr: 0.9500
    ndcg: 0.9312
    hallucination_rate: 0.1250
```

Results are saved to `eval/results.json`.

## Roadmap

### Week 1 — Baseline RAG
- [x] Document ingestion, chunking, embedding
- [x] FAISS vector store and dense retrieval
- [x] LLM generation with OpenAI/Ollama
- [x] FastAPI `/ask` endpoint

### Week 2 — Hybrid Retrieval & Citations
- [x] BM25 sparse retrieval (`rank-bm25`)
- [x] Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- [x] Source citations with bracket notation in LLM answers
- [x] Improved context formatting in prompts

### Week 3 — Quality Improvements
- [x] Cross-encoder reranker
- [x] Query rewriting
- [x] Better prompts
- [x] Answer abstention (refuse when context is insufficient)

### Week 4 — Portfolio Ready
- [x] Evaluation dataset and benchmark suite
- [x] Unit tests (pytest)
- [x] Docker and docker-compose
- [x] Architecture diagram (Mermaid)
- [x] Comprehensive README

### Week 5 — Production Hardening
- [x] Pinned dependencies for reproducible builds
- [x] Multi-stage Docker build with non-root user
- [x] API key authentication (`X-API-Key` header)
- [x] Rate limiting (slowapi, configurable window)
- [x] Lifespan context manager (replaced deprecated startup events)
- [x] Full test coverage for pipeline, prompting, and retriever (67 tests)
- [x] CVE scanning and remediation (0 critical/high, separated dev dependencies)
- [x] Mathematical evaluation framework (RAGAS, BERTScore, NDCG)
- [x] Observability (OpenTelemetry tracing, Prometheus metrics)

### Future
- [ ] Streaming responses
- [ ] Multi-modal document support (PDF, HTML)
- [ ] CI pipeline (GitHub Actions)
- [ ] Configurable chunking strategies
- [ ] Web UI
- [ ] Auto-scaling (Kubernetes + HPA)
- [ ] Advanced RAG (HyDE, semantic caching, guardrails)
