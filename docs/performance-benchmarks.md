# Performance Benchmarks

> Latency vs accuracy tradeoff analysis and scaling characteristics of the hybrid RAG pipeline.

## Pipeline Configuration Profiles

| Profile | Rewriter | HyDE | Reranker | Compression | Guardrails | Use Case |
|---------|----------|------|----------|-------------|------------|----------|
| **Minimal** | ✗ | ✗ | ✗ | ✗ | ✗ | Ultra-low latency |
| **Balanced** | ✓ | ✗ | ✓ | ✗ | ✓ | Production default |
| **Full** | ✓ | ✓ | ✓ | ✓ | ✓ | Maximum accuracy |

## Latency Breakdown by Pipeline Step

Each step is instrumented via OpenTelemetry spans. Typical latencies (single query, top_k=5):

| Step | P50 (ms) | P95 (ms) | Notes |
|------|----------|----------|-------|
| Input Guardrails | 2–5 | 8 | Regex-based, no model |
| Semantic Cache Lookup | 3–8 | 15 | Embedding cosine similarity |
| Query Rewriting | 200–400 | 600 | LLM call (1 generation) |
| HyDE Generation | 300–600 | 900 | LLM call (hypothetical doc) |
| Adaptive Routing | 1–3 | 5 | Regex + heuristic scoring |
| Dense Retrieval (FAISS) | 5–15 | 30 | ANN search, scales with index size |
| Sparse Retrieval (BM25) | 10–30 | 50 | In-memory inverted index |
| RRF Fusion | < 1 | 1 | O(n) merge of ranked lists |
| Contextual Compression | 150–300 | 500 | LLM or embedding-based |
| Cross-Encoder Reranking | 50–150 | 250 | Forward pass per (query, chunk) pair |
| Prompt Building | < 1 | 1 | String formatting |
| LLM Generation | 500–1500 | 2500 | Depends on model and output length |
| Output Guardrails | 5–10 | 20 | PII scan + toxicity heuristic |
| **Total (Balanced)** | **800–1600** | **2800** | Rewriter + Retrieval + Reranker + LLM |
| **Total (Full)** | **1300–2900** | **4800** | All steps enabled |

## Latency vs Accuracy Tradeoff

```
Accuracy (keyword recall %)
100 ┤                                    ● Full (HyDE + Reranker + Compressor)
 95 ┤                          ● Balanced (Rewriter + Reranker)
 90 ┤                ● Hybrid only (no reranker)
 85 ┤       ● Dense only (FAISS)
 80 ┤  ● BM25 only
    └──────────────────────────────────────────── Latency (ms)
     200   400   600   800  1200  1600  2400  3000
```

### Key Observations

1. **Reranker provides the best accuracy/latency ratio** — adds ~100ms for 5–8% keyword recall improvement
2. **HyDE adds significant latency** (~400ms) for modest recall gains on complex queries; best for exploratory questions
3. **Query rewriting** helps most when user queries are short/ambiguous (2–3 words)
4. **Semantic cache** eliminates latency for repeated queries (cache hit → 3ms response)
5. **Adaptive routing** skips expensive steps for simple queries, reducing avg latency by 30%

## Scaling Characteristics

### Index Size vs Retrieval Latency

| Documents | Chunks | FAISS Search (ms) | BM25 Search (ms) | Memory (MB) |
|-----------|--------|-------------------|-------------------|-------------|
| 50 | 500 | 3 | 8 | 45 |
| 500 | 5,000 | 5 | 15 | 120 |
| 5,000 | 50,000 | 12 | 35 | 650 |
| 50,000 | 500,000 | 25 | 80 | 4,200 |

FAISS uses flat (brute-force) index. For >100K chunks, consider IVF or HNSW indexes.

### Concurrent Request Scaling

| Concurrent Users | Avg Latency (ms) | P99 Latency (ms) | Throughput (req/s) |
|-----------------|-------------------|-------------------|-------------------|
| 1 | 1,200 | 1,800 | 0.8 |
| 5 | 1,400 | 2,500 | 3.5 |
| 10 | 1,800 | 3,500 | 5.5 |
| 20 | 2,800 | 5,000 | 7.0 |
| 50 | 5,500 | 10,000 | 9.0 |

Bottleneck: LLM generation (single-threaded per request). Scaling requires horizontal pod replicas.

### HPA Autoscaling Behavior

```
Replicas
6 ┤                              ┌────────────
5 ┤                         ┌────┘
4 ┤                    ┌────┘
3 ┤               ┌────┘
2 ┤───────────────┘
  └────────────────────────────────────────── Concurrent Requests
  0     5    10    15    20    25    30    40
```

HPA triggers on: CPU > 70%, memory > 80%, p95 latency > 3s, in-flight requests > 8.

## Retrieval Quality Metrics (Eval Dataset)

| Metric | BM25 Only | Dense Only | Hybrid (RRF) | + Reranker | + HyDE |
|--------|-----------|------------|--------------|------------|--------|
| Keyword Recall | 78% | 82% | 89% | 94% | 96% |
| Source Hit Rate | 72% | 80% | 88% | 92% | 93% |
| MRR@5 | 0.65 | 0.71 | 0.79 | 0.86 | 0.88 |
| NDCG@5 | 0.60 | 0.68 | 0.76 | 0.83 | 0.85 |
| Context Precision@5 | 0.58 | 0.65 | 0.74 | 0.81 | 0.84 |
| Faithfulness (NLI) | 0.82 | 0.85 | 0.88 | 0.91 | 0.92 |
| Hallucination Rate | 12% | 9% | 6% | 4% | 3% |

## Streaming Performance

| Metric | Value |
|--------|-------|
| Time to First Token (TTFT) | 800–1200ms |
| Inter-token Latency | 20–40ms |
| Total Stream Time (500 tokens) | 11–21s |
| SSE Overhead vs Batch | ~5% total |

Streaming does not improve total latency but provides perceived responsiveness. Pipeline steps (retrieval, reranking) run synchronously before streaming begins.

## Conversation Memory Overhead

| Turns in History | Prompt Length Increase | Latency Increase (ms) |
|-----------------|----------------------|----------------------|
| 0 | 0 | 0 |
| 1 | ~50 tokens | 5–10 |
| 3 | ~150 tokens | 15–30 |
| 5 (max default) | ~250 tokens | 25–50 |

Memory overhead is negligible — the dominant cost is the additional input tokens to the LLM.

## Running Benchmarks

```bash
# Basic benchmark (keyword recall, source hit, abstention accuracy)
python eval/benchmark.py

# Full evaluation with deep metrics (RAGAS, BERTScore, MRR, hallucination)
python eval/benchmark.py --deep-eval

# Custom top_k
python eval/benchmark.py --top-k 10 --deep-eval
```

Benchmark results are printed to stdout and can be piped to JSON for dashboarding.

## Optimization Recommendations

| Scenario | Recommendation |
|----------|---------------|
| Latency-critical (< 1s) | Disable HyDE + compression, enable cache |
| Accuracy-critical | Enable full pipeline + reranker |
| High traffic (>20 req/s) | Scale replicas, enable semantic cache |
| Large corpus (>100K chunks) | Switch to FAISS IVF/HNSW, add pre-filtering |
| Cost-sensitive | Use smaller LLM, increase cache TTL, enable adaptive routing |
