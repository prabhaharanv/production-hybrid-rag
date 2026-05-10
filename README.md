# Production Hybrid RAG

A production-grade Retrieval-Augmented Generation system with hybrid search, cross-encoder reranking, query rewriting, source citations, and answer abstention.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.7.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-274%20passed-brightgreen)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram.

```
Question → API Key Auth → Rate Limiter → Query Rewriting → Hybrid Retrieval (FAISS + BM25 → RRF) → Cross-Encoder Reranking → LLM Generation → Abstention Check → Citation Extraction → Response
```

## Features

- **Ingestion**: Document loading (`.txt`, `.md`, `.pdf`, `.html`, `.docx`), configurable chunking strategies
- **Multi-modal documents**: PDF (PyMuPDF), HTML (BeautifulSoup4), DOCX (python-docx)
- **Chunking strategies**: Word, sentence, recursive, token-aware — strategy pattern with config toggle
- **Embeddings**: Sentence-transformer (`all-MiniLM-L6-v2`) with FAISS indexing
- **Sparse retrieval**: BM25 keyword search (`rank-bm25`)
- **Hybrid retrieval**: Reciprocal Rank Fusion (RRF) merging dense + sparse results
- **Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) for precision
- **Query rewriting**: LLM rewrites vague queries before retrieval
- **Source citations**: Bracket notation `[1]`, `[2]` with structured citation objects
- **Answer abstention**: Refuses to answer when context is insufficient
- **LLM generation**: OpenAI or Ollama (local)
- **Streaming**: SSE endpoint (`/ask/stream`) for real-time token-by-token responses
- **API**: FastAPI with Pydantic validation, lifespan-managed startup
- **API security**: API key authentication (`X-API-Key` header) and rate limiting (slowapi)
- **Web UI**: Streamlit app with streaming responses, source panel, confidence indicators
- **Docker**: Multi-stage build, non-root user, pinned dependencies, CVE-scanned (0 critical/high)
- **CI/CD**: GitHub Actions pipeline — lint → test → build Docker → push to GHCR
- **Observability**: OpenTelemetry tracing, Prometheus metrics, structlog JSON logging with correlation IDs
- **Monitoring**: Grafana dashboards (latency percentiles, error rates, abstention rate), Prometheus alerting rules
- **Health checks**: Liveness (`/health`) and deep readiness probes (`/health/ready`) for Kubernetes/load balancers
- **Guardrails**: PII detection, prompt injection defense, output toxicity filtering
- **HyDE retrieval**: Hypothetical Document Embeddings for improved recall
- **Semantic caching**: Embedding-similarity cache with TTL to reduce LLM calls
- **Adaptive retrieval**: Routes queries to BM25-only, dense-only, or full hybrid based on complexity
- **Contextual compression**: LLM or embedding-based extraction of relevant sentences from chunks
- **Parent-child chunking**: Small chunks for retrieval precision, full parent chunks for generation context
- **Evaluation**: Benchmark suite with keyword recall, source hit rate, and abstention accuracy
- **Deep evaluation**: Mathematical eval framework — RAGAS (faithfulness, answer relevance, context precision/recall), BERTScore, MRR, NDCG@K, NLI-based hallucination detection
- **Kubernetes**: Deployment, Service, HPA, ConfigMap, Secrets, PDB, PVCs
- **Helm chart**: Parameterized chart with dev/staging/prod value overrides
- **Auto-scaling**: HPA on CPU/memory + custom metrics (p95 latency, in-flight requests via prometheus-adapter)
- **Load testing**: Locust scripts simulating realistic traffic patterns
- **Testing**: 274 unit tests across 20 test files (pytest)

## Project Structure

```
production-hybrid-rag/
├── app/
│   ├── api.py              # FastAPI application — /ask, /ask/stream (SSE), /health
│   ├── config.py           # Settings loaded from .env
│   ├── schemas.py          # Request/response Pydantic models
│   └── observability/      # Observability & monitoring
│       ├── tracing.py      # OpenTelemetry distributed tracing
│       ├── metrics.py      # Prometheus metrics and counters
│       ├── logging.py      # structlog JSON logging with correlation IDs
│       └── health.py       # Liveness and readiness health checks
├── rag/
│   ├── loader.py           # Multi-modal loader (.txt, .md, .pdf, .html, .docx)
│   ├── chunking.py         # Configurable chunking (word, sentence, recursive, token)
│   ├── embeddings.py       # Sentence-transformer wrapper
│   ├── vector_store.py     # FAISS index save/load/search
│   ├── bm25_retriever.py   # BM25 sparse index and search
│   ├── retriever.py        # Dense, Sparse, Hybrid retrievers + RRF
│   ├── reranker.py         # Cross-encoder reranker
│   ├── query_rewriter.py   # LLM-based query rewriting
│   ├── prompting.py        # Prompt with citation + abstention rules
│   ├── generator.py        # LLM generation + streaming (OpenAI-compatible)
│   ├── pipeline.py         # Rewrite → Retrieve → Rerank → Generate (sync + stream)
│   ├── ingest.py           # End-to-end ingestion orchestration
│   ├── hyde.py             # Hypothetical Document Embeddings (HyDE)
│   ├── cache.py            # Semantic caching with embedding similarity
│   ├── guardrails.py       # PII detection, prompt injection, toxicity filtering
│   ├── adaptive.py         # Adaptive retrieval routing by query complexity
│   ├── compressor.py       # Contextual compression (LLM + embedding-based)
│   ├── parent_child.py     # Parent-child chunking strategy
│   ├── memory.py           # Conversation memory (multi-turn sliding window)
│   ├── documents.py        # Document management (upload, list, delete)
│   └── ab_testing.py       # A/B testing framework for retrieval strategies
├── ui/
│   ├── app.py              # Streamlit Web UI (streaming, sources, confidence)
│   └── requirements.txt
├── eval/
│   ├── dataset.json        # Evaluation questions with expected answers + ground truth
│   ├── metrics.py          # Mathematical evaluation metrics
│   └── benchmark.py        # Benchmark runner with metrics
├── tests/                  # Unit tests (pytest)
├── scripts/
│   └── ingest_docs.py      # CLI ingestion entry point
├── docs/
│   └── architecture.md     # Mermaid architecture diagram
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: lint → test → build → push
├── data/
│   └── raw/                # Place source documents here (.txt, .md, .pdf, .html, .docx)
├── k8s/                    # Raw Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment.yaml     # Health probes, resource limits, PVC mounts
│   ├── service.yaml
│   ├── hpa.yaml            # CPU/memory + custom metrics (p95 latency, in-flight)
│   ├── pdb.yaml            # Pod disruption budget
│   ├── pvc.yaml            # Data + HuggingFace model cache
│   └── prometheus-adapter.yaml
├── helm/rag/               # Parameterized Helm chart
│   ├── Chart.yaml
│   ├── values.yaml         # Defaults
│   ├── values-dev.yaml     # Dev: 1 replica, no HPA, debug logs
│   ├── values-staging.yaml # Staging: 2-5 replicas, CPU/memory HPA
│   ├── values-prod.yaml    # Prod: 3-20 replicas, custom metrics HPA
│   └── templates/
├── loadtest/
│   ├── locustfile.py       # Locust load test scenarios
│   └── requirements.txt
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
#    Supports .txt, .md, .pdf, .html, .docx

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

# Chunking strategy: word | sentence | recursive | token
CHUNKING_STRATEGY=word
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

### `POST /ask/stream` (SSE)

Streams the response token-by-token via Server-Sent Events:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 3}'
```

SSE events:
```
data: {"event": "metadata", "rewritten_query": "...", "retrieved_chunks": [...]}

data: {"event": "token", "data": "Retrieval"}
data: {"event": "token", "data": "-Augmented"}
data: {"event": "token", "data": " Generation"}
...

data: {"event": "done", "abstained": false, "citations": [...]}
```

## Web UI

A Streamlit interface with real-time streaming, source panel, and confidence indicators:

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Open http://localhost:8501. The UI connects to the RAG API (default: `http://localhost:8000`).

| Feature | Description |
|---------|-------------|
| Streaming | Tokens appear in real-time as the LLM generates |
| Source panel | Sidebar shows all retrieved chunks with relevance scores |
| Confidence | High/Medium/Low based on average retrieval score |
| Citations | Cited sources are highlighted in the source panel |

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

## Kubernetes Deployment

### Raw manifests

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Check rollout
kubectl -n rag get pods,hpa
```

### Helm chart

```bash
# Dev
helm install rag helm/rag -n rag --create-namespace \
  -f helm/rag/values-dev.yaml \
  --set secrets.OPENAI_API_KEY=your-key

# Staging
helm install rag helm/rag -n rag --create-namespace \
  -f helm/rag/values-staging.yaml \
  --set secrets.OPENAI_API_KEY=your-key

# Production (with custom metrics HPA)
helm install rag helm/rag -n rag --create-namespace \
  -f helm/rag/values-prod.yaml \
  --set secrets.OPENAI_API_KEY=your-key
```

### Auto-scaling

The HPA scales on four signals:

| Signal | Type | Target | Notes |
|--------|------|--------|-------|
| CPU utilisation | Resource | 70% | Built-in metrics-server |
| Memory utilisation | Resource | 80% | Built-in metrics-server |
| p95 request latency | Custom | 5 s | Requires prometheus-adapter |
| In-flight requests | Custom | 10/pod | Requires prometheus-adapter |

To enable custom metrics, install [prometheus-adapter](https://github.com/kubernetes-sigs/prometheus-adapter) with the provided config:

```bash
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  -f k8s/prometheus-adapter.yaml -n monitoring
```

Scale-up is aggressive (50% increase/min), scale-down is conservative (25% decrease per 2 min with 5 min stabilisation window) to prevent flapping.

## Load Testing

```bash
pip install -r loadtest/requirements.txt

# Web UI at http://localhost:8089
locust -f loadtest/locustfile.py --host http://localhost:8000

# Headless quick smoke test (10 users, 60 seconds)
locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 10 -r 2 -t 60s

# Sustained load test (100 users, 5 minutes)
locust -f loadtest/locustfile.py --host http://localhost:8000 \
    --headless -u 100 -r 10 -t 5m

# With API key
RAG_API_KEY=your-key locust -f loadtest/locustfile.py --host http://localhost:8000
```

Traffic mix: 43% `/ask` (in-scope), 13% `/ask` (custom top_k), 9% `/ask` (out-of-scope), 22% `/health`, 9% `/health/ready`, 4% `/metrics`.

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

## Chunking Strategies

Choose a strategy via the `CHUNKING_STRATEGY` environment variable or pass `strategy=` to `chunk_documents()`:

| Strategy | Description | Best for |
|----------|-------------|----------|
| `word` | Fixed-size word-level chunks with overlap (default) | General use, fast |
| `sentence` | Groups sentences up to chunk_size words, respects boundaries | Prose documents |
| `recursive` | Hierarchical: paragraphs → sentences → words | Structured docs (markdown, reports) |
| `token` | BPE token-aware via tiktoken (`cl100k_base`) | LLM token budget alignment |

```bash
# Set globally
export CHUNKING_STRATEGY=sentence

# Or per-ingestion
CHUNKING_STRATEGY=recursive python scripts/ingest_docs.py
```

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):

```
push/PR to main → lint (ruff) → test (pytest) → build Docker → push to GHCR
```

- **Lint**: `ruff check` + `ruff format --check` for consistent code style
- **Test**: Full pytest suite on Python 3.12
- **Build**: Multi-stage Docker build with layer caching (GitHub Actions cache)
- **Push**: Container image pushed to GitHub Container Registry (`ghcr.io`) on merge to `main`

## Roadmap

### Baseline RAG
- [x] Document ingestion, chunking, embedding
- [x] FAISS vector store and dense retrieval
- [x] LLM generation with OpenAI/Ollama
- [x] FastAPI `/ask` endpoint

### Hybrid Retrieval & Citations
- [x] BM25 sparse retrieval (`rank-bm25`)
- [x] Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- [x] Source citations with bracket notation in LLM answers
- [x] Improved context formatting in prompts

### Quality Improvements
- [x] Cross-encoder reranker
- [x] Query rewriting
- [x] Better prompts
- [x] Answer abstention (refuse when context is insufficient)

### Evaluation & Testing
- [x] Evaluation dataset and benchmark suite
- [x] Unit tests (pytest)
- [x] Docker and docker-compose
- [x] Architecture diagram (Mermaid)
- [x] Mathematical evaluation framework (RAGAS, BERTScore, NDCG)

### Production Hardening
- [x] Pinned dependencies for reproducible builds
- [x] Multi-stage Docker build with non-root user
- [x] API key authentication (`X-API-Key` header)
- [x] Rate limiting (slowapi, configurable window)
- [x] CVE scanning and remediation (0 critical/high)
- [x] Observability (OpenTelemetry tracing, Prometheus metrics, structlog)
- [x] Grafana dashboards and Prometheus alerting rules

### Auto-Scaling Infrastructure
- [x] Kubernetes manifests (Deployment, Service, HPA, ConfigMap, Secrets, PDB)
- [x] Horizontal Pod Autoscaler (CPU/memory + custom metrics via prometheus-adapter)
- [x] Helm chart with dev/staging/prod value overrides
- [x] Locust load testing scripts

### Streaming & Multi-Modal
- [x] SSE streaming endpoint (`/ask/stream`) — real-time token delivery
- [x] Multi-modal document support (PDF, HTML, DOCX)
- [x] Configurable chunking strategies (word, sentence, recursive, token)
- [x] CI/CD pipeline (GitHub Actions: lint → test → build → push)
- [x] Web UI (Streamlit with streaming, source panel, confidence indicators)

### Advanced RAG
- [x] HyDE (Hypothetical Document Embeddings) for improved recall
- [x] Semantic caching with embedding similarity and TTL
- [x] Guardrails (PII detection, prompt injection defense, output toxicity filtering)
- [x] Adaptive retrieval routing (simple → BM25, complex → full hybrid)
- [x] Contextual compression (LLM-based and embedding-based)
- [x] Parent-child chunking (small chunks for search, large for generation)

### Conversation & Management
- [x] Conversation memory (multi-turn context with sliding window)
- [x] Document management API (upload, list, delete)
- [x] A/B testing framework for retrieval strategies
