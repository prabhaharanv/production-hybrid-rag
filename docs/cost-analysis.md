# Cost Analysis

> Token usage tracking, cost per query, and optimization strategies for the hybrid RAG pipeline.

## Token Flow Per Query

A single `/ask` request consumes tokens at multiple pipeline stages:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage              │ Input Tokens    │ Output Tokens │ Model     │
├────────────────────┼─────────────────┼───────────────┼───────────┤
│ Query Rewriting    │ ~100            │ ~30           │ LLM       │
│ HyDE Generation   │ ~120            │ ~150          │ LLM       │
│ Final Generation   │ ~800–2000       │ ~200–500      │ LLM       │
│ Contextual Compress│ ~1500           │ ~300          │ LLM       │
└────────────────────┴─────────────────┴───────────────┴───────────┘
```

### Breakdown by Configuration Profile

| Profile | Input Tokens/Query | Output Tokens/Query | Total Tokens |
|---------|-------------------|--------------------:|-------------:|
| **Minimal** (no rewriter, no HyDE) | 800 | 300 | 1,100 |
| **Balanced** (rewriter, no HyDE) | 900 | 330 | 1,230 |
| **Full** (rewriter + HyDE + compression) | 2,500 | 780 | 3,280 |

## Cost Per Query (USD)

### OpenAI Pricing (as of 2024)

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Cost/Query (Balanced) | Cost/Query (Full) |
|-------|--------------------:|---------------------:|----------------------:|------------------:|
| gpt-4.1-mini | $0.40 | $1.60 | $0.0009 | $0.0024 |
| gpt-4.1 | $2.00 | $8.00 | $0.0044 | $0.0120 |
| gpt-4o | $2.50 | $10.00 | $0.0051 | $0.0148 |
| gpt-3.5-turbo | $0.50 | $1.50 | $0.0009 | $0.0022 |

### Ollama / Self-hosted (Local Inference)

| Model | VRAM Required | Cost/Query | Notes |
|-------|--------------|------------|-------|
| llama3.2:3b | 4 GB | $0.00 | Free, lowest quality |
| llama3.1:8b | 8 GB | $0.00 | Good quality/cost tradeoff |
| llama3.1:70b | 48 GB | $0.00 | Near-GPT-4 quality |

Infrastructure cost for self-hosted:
- **Development**: $0 (local GPU or CPU inference with Ollama)
- **Production (1x A10G)**: ~$1.00/hr → $0.0003/query at 10 req/s
- **Production (1x A100)**: ~$3.50/hr → $0.0001/query at 30 req/s

## Monthly Cost Projections

### Cloud LLM (gpt-4.1-mini, Balanced profile)

| Daily Queries | Monthly Tokens | Monthly Cost | Cost with Cache (40% hit) |
|--------------|---------------:|-------------:|--------------------------:|
| 100 | 3.7M | $2.70 | $1.62 |
| 1,000 | 37M | $27.00 | $16.20 |
| 10,000 | 370M | $270.00 | $162.00 |
| 100,000 | 3.7B | $2,700.00 | $1,620.00 |

### Infrastructure Costs (AWS)

| Component | Instance Type | Monthly Cost | Purpose |
|-----------|--------------|-------------:|---------|
| RAG API (2 replicas) | t3.medium | $60 | FastAPI + retrieval |
| FAISS index in-memory | r6i.large | $98 | Vector search |
| Redis (cache) | cache.t3.micro | $13 | Semantic cache |
| Prometheus + Grafana | t3.small | $15 | Monitoring |
| OTel Collector | t3.micro | $8 | Trace collection |
| **Total infra** | | **$194/mo** | |

## Token Usage Tracking

The pipeline tracks approximate token usage per request:

```python
# In pipeline.py — returned in every response
"token_usage": {
    "prompt_tokens": 920,      # estimated (1 token ≈ 4 chars)
    "completion_tokens": 285,
    "total_tokens": 1205
}
```

### Monitoring Token Usage

Exposed via Prometheus metrics at `GET /metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `rag_tokens_total{type="prompt"}` | Counter | Total input tokens consumed |
| `rag_tokens_total{type="completion"}` | Counter | Total output tokens consumed |
| `rag_cost_usd_total` | Counter | Estimated cumulative cost |
| `rag_query_tokens_histogram` | Histogram | Token distribution per query |

Grafana dashboard: cost/hour, cost/day, tokens by pipeline step.

## Cost Optimization Strategies

### 1. Semantic Cache (Biggest Impact)

| Cache Hit Rate | Token Savings | Cost Reduction |
|---------------|--------------|----------------|
| 20% | 20% | 20% |
| 40% | 40% | 40% |
| 60% | 60% | 60% |

Configure: `CACHE_SIMILARITY_THRESHOLD=0.92`, `CACHE_TTL=3600`

Typical hit rates:
- Internal knowledge base (repeated questions): 40–60%
- Customer-facing chatbot: 20–35%
- Research tool (unique queries): 5–15%

### 2. Adaptive Routing

Simple queries (factoid lookups) skip HyDE and compression, saving ~2,000 tokens/query.
Complex queries still get the full pipeline.

Estimated savings: **25–35% token reduction** on mixed workloads.

### 3. Model Selection

| Strategy | Token Cost Impact | Quality Impact |
|----------|------------------|----------------|
| Use gpt-4.1-mini for rewriting | −60% on rewrite step | Minimal |
| Use gpt-4.1-mini for generation | −80% vs gpt-4o | ~5% quality drop |
| Local Ollama for dev/test | −100% | Varies by model |

### 4. Contextual Compression

Reduces prompt size by 40–60% before final generation, but adds a compression LLM call.
Net savings only when:  `compression_token_cost < generation_savings`

**Break-even**: When retrieved context is > 1,500 tokens (typically >3 chunks).

### 5. Prompt Engineering

| Technique | Savings |
|-----------|---------|
| Shorter system prompt | 50–100 tokens |
| Limit context to top-3 chunks | ~400 tokens |
| Truncate chunk text to 200 chars | ~300 tokens |
| Skip citations in prompt | ~100 tokens |

### 6. Conversation Memory Impact

| Max Turns | Avg Extra Tokens | Monthly Cost Increase (10K queries/day) |
|-----------|-----------------|----------------------------------------|
| 0 | 0 | $0 |
| 3 | 150 | +$18/mo |
| 5 | 250 | +$30/mo |
| 10 | 500 | +$60/mo |

Default `MEMORY_MAX_TURNS=5` balances context quality vs cost.

## Cost Alerting

Set Prometheus alerts:

```yaml
# Alert if hourly cost exceeds threshold
- alert: HighTokenCost
  expr: rate(rag_cost_usd_total[1h]) * 3600 > 5.00
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "RAG token cost exceeding $5/hour"
```

## Summary: Cost Per Query by Scenario

| Scenario | Model | Profile | Cache | Cost/Query |
|----------|-------|---------|-------|-----------|
| Dev/testing | Ollama llama3.2:3b | Minimal | No | $0.000 |
| Startup MVP | gpt-4.1-mini | Balanced | Yes (40%) | $0.0005 |
| Production | gpt-4.1-mini | Balanced | Yes (40%) | $0.0005 |
| Enterprise | gpt-4.1 | Full | Yes (30%) | $0.0084 |
| High-accuracy | gpt-4o | Full | Yes (30%) | $0.0104 |
