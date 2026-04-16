# Production Hybrid RAG

A production-grade Retrieval-Augmented Generation system with hybrid search, cross-encoder reranking, query rewriting, source citations, and answer abstention.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.3.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram.

```
Question → Query Rewriting → Hybrid Retrieval (FAISS + BM25 → RRF) → Cross-Encoder Reranking → LLM Generation → Abstention Check → Citation Extraction → Response
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
- **API**: FastAPI with Pydantic validation
- **Docker**: Single-command deployment
- **Evaluation**: Benchmark suite with keyword recall, source hit rate, and abstention accuracy

## Project Structure

```
production-hybrid-rag/
├── app/
│   ├── api.py              # FastAPI application and endpoints
│   ├── config.py           # Settings loaded from .env
│   └── schemas.py          # Request/response Pydantic models
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
│   ├── dataset.json        # Evaluation questions with expected answers
│   └── benchmark.py        # Benchmark runner with metrics
├── tests/                  # Unit tests (pytest)
├── scripts/
│   └── ingest_docs.py      # CLI ingestion entry point
├── docs/
│   └── architecture.md     # Mermaid architecture diagram
├── data/
│   └── raw/                # Place source documents here
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

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
```

## API

### `POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
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
# Build and run
docker compose up --build

# Or build manually
docker build -t hybrid-rag .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data hybrid-rag
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_pipeline.py -v
```

## Evaluation

```bash
# Run the benchmark suite (requires ingested data + LLM access)
python eval/benchmark.py

# Use a custom dataset
python eval/benchmark.py path/to/dataset.json
```

Metrics reported:
- **Keyword recall**: Fraction of expected keywords found in the answer
- **Source hit rate**: Whether the correct source document was cited
- **Abstention accuracy**: Whether the system correctly refused out-of-scope questions
- **Latency**: End-to-end time per question

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

### Future
- [ ] Streaming responses
- [ ] Multi-modal document support (PDF, HTML)
- [ ] CI pipeline (GitHub Actions)
- [ ] Configurable chunking strategies
- [ ] Web UI
