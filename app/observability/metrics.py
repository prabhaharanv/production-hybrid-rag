"""
Prometheus Metrics.

Tracks: request rate, latency p50/p95/p99, retrieval scores,
abstention rate, and token usage.
Exposes a ``/metrics`` endpoint for Prometheus scraping.
"""

import time
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


class RAGMetrics:
    """Container for all Prometheus metrics used by the RAG service."""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()

        # ── Request-level ──
        self.request_count = Counter(
            "rag_requests_total",
            "Total /ask requests",
            labelnames=["status"],
            registry=self.registry,
        )

        self.request_latency = Histogram(
            "rag_request_latency_seconds",
            "End-to-end request latency",
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry,
        )

        # ── Pipeline step latencies ──
        self.step_latency = Histogram(
            "rag_step_latency_seconds",
            "Latency per pipeline step",
            labelnames=["step"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        # ── Retrieval ──
        self.retrieval_score = Histogram(
            "rag_retrieval_top_score",
            "Top retrieval score per request",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

        self.chunks_retrieved = Histogram(
            "rag_chunks_retrieved",
            "Number of chunks returned by retriever",
            buckets=(1, 2, 3, 5, 10, 15, 20),
            registry=self.registry,
        )

        # ── Abstention ──
        self.abstention_count = Counter(
            "rag_abstentions_total",
            "Total abstained responses",
            registry=self.registry,
        )

        # ── Token usage (approximation) ──
        self.token_usage = Counter(
            "rag_token_usage_total",
            "Approximate token usage",
            labelnames=["type"],
            registry=self.registry,
        )

        # ── In-flight requests (for HPA custom metric) ──
        self.requests_in_flight = Gauge(
            "rag_requests_in_flight",
            "Number of /ask requests currently being processed",
            registry=self.registry,
        )

        # ── Liveness gauge ──
        self.pipeline_ready = Gauge(
            "rag_pipeline_ready",
            "1 if pipeline is initialised and healthy",
            registry=self.registry,
        )

    def generate_latest(self) -> bytes:
        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


# ── Module-level singleton ──
_metrics: RAGMetrics | None = None


def init_metrics(registry: CollectorRegistry | None = None) -> RAGMetrics:
    global _metrics
    _metrics = RAGMetrics(registry=registry)
    return _metrics


def get_metrics() -> RAGMetrics:
    global _metrics
    if _metrics is None:
        _metrics = RAGMetrics()
    return _metrics


@contextmanager
def track_request():
    """Context manager that times an entire /ask request."""
    m = get_metrics()
    m.requests_in_flight.inc()
    start = time.perf_counter()
    try:
        yield m
        m.request_count.labels(status="success").inc()
    except Exception:
        m.request_count.labels(status="error").inc()
        raise
    finally:
        m.request_latency.observe(time.perf_counter() - start)
        m.requests_in_flight.dec()


@contextmanager
def track_step(step_name: str):
    """Context manager that records latency for a single pipeline step."""
    m = get_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        m.step_latency.labels(step=step_name).observe(time.perf_counter() - start)
