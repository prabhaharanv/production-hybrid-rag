# Production Hybrid RAG

Baseline RAG system: ingest documents, chunk, embed, index, and answer questions via a FastAPI endpoint.

## Features

- Document ingestion (`.txt`, `.md`)
- Character-level chunking with overlap
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- FAISS dense vector search
- BM25 sparse keyword retrieval
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Source citations with bracket notation `[1]`, `[2]` in answers
- LLM answer generation (OpenAI or Ollama)
- FastAPI `/ask` endpoint with retrieved context and structured citations

## Project Structure

```
production-hybrid-rag/
├── app/
│   ├── api.py          # FastAPI application and endpoints
│   ├── config.py       # Settings loaded from .env
│   └── schemas.py      # Request/response models
├── rag/
│   ├── loader.py       # Document loader (.txt, .md)
│   ├── chunking.py     # Text chunking with overlap
│   ├── embeddings.py   # Sentence-transformer wrapper
│   ├── vector_store.py # FAISS index save/load/search
│   ├── bm25_retriever.py # BM25 sparse index and search
│   ├── retriever.py    # Dense, Sparse, and Hybrid retrievers with RRF
│   ├── prompting.py    # RAG prompt with citation instructions
│   ├── generator.py    # LLM generation (OpenAI-compatible)
│   ├── pipeline.py     # Retriever → Generator pipeline
│   └── ingest.py       # End-to-end ingestion orchestration
├── scripts/
│   └── ingest_docs.py  # CLI ingestion entry point
├── data/
│   └── raw/            # Place source documents here
├── .env.example        # Environment variable template
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
  "answer": "...",
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
- [ ] Cross-encoder reranker
- [ ] Query rewriting
- [ ] Better prompts
- [ ] Answer abstention (refuse when context is insufficient)

### Future
- [ ] Evaluation dataset and benchmarks
- [ ] Tests and CI
- [ ] Docker
- [ ] Architecture diagram
