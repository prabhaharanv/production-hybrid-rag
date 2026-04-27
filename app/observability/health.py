"""
Health Checks (Liveness + Readiness Probes).

Checks: model loaded? Index loaded? LLM reachable?
Returns structured status for Kubernetes probes or load-balancer health checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ComponentStatus:
    name: str
    healthy: bool
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthChecker:
    """Aggregates liveness / readiness status for the RAG service."""

    pipeline: object | None = None
    _checks: list = field(default_factory=list)

    def set_pipeline(self, pipeline: object | None) -> None:
        self.pipeline = pipeline

    # ── Liveness ──

    def liveness(self) -> dict:
        """Lightweight probe: is the process alive and serving?"""
        return {"status": "ok"}

    # ── Readiness ──

    def readiness(self) -> dict:
        """Deep probe: are all components operational?"""
        components = [
            self._check_pipeline(),
            self._check_retriever(),
            self._check_generator(),
            self._check_index(),
        ]
        healthy = all(c.healthy for c in components)
        return {
            "status": "ready" if healthy else "degraded",
            "healthy": healthy,
            "components": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "detail": c.detail,
                    "latency_ms": round(c.latency_ms, 2),
                }
                for c in components
            ],
        }

    # ── Individual checks ──

    def _check_pipeline(self) -> ComponentStatus:
        if self.pipeline is None:
            return ComponentStatus("pipeline", False, "not initialised")
        return ComponentStatus("pipeline", True, "initialised")

    def _check_retriever(self) -> ComponentStatus:
        if self.pipeline is None:
            return ComponentStatus("retriever", False, "pipeline not initialised")
        retriever = getattr(self.pipeline, "retriever", None)
        if retriever is None:
            return ComponentStatus("retriever", False, "no retriever configured")
        start = time.perf_counter()
        try:
            # Probe with a trivial query — just verify it doesn't crash
            retriever.retrieve("health check probe", top_k=1)
            elapsed = (time.perf_counter() - start) * 1000
            return ComponentStatus("retriever", True, "ok", elapsed)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return ComponentStatus("retriever", False, str(exc), elapsed)

    def _check_generator(self) -> ComponentStatus:
        if self.pipeline is None:
            return ComponentStatus("generator", False, "pipeline not initialised")
        generator = getattr(self.pipeline, "generator", None)
        if generator is None:
            return ComponentStatus("generator", False, "no generator configured")
        # Check that the client object exists (don't make an actual LLM call)
        client = getattr(generator, "client", None)
        if client is None:
            return ComponentStatus("generator", False, "OpenAI client not initialised")
        return ComponentStatus("generator", True, "client initialised")

    def _check_index(self) -> ComponentStatus:
        if self.pipeline is None:
            return ComponentStatus("index", False, "pipeline not initialised")
        retriever = getattr(self.pipeline, "retriever", None)
        if retriever is None:
            return ComponentStatus("index", False, "no retriever")
        # For HybridRetriever, check both dense and sparse stores
        dense = getattr(retriever, "dense", None)
        sparse = getattr(retriever, "sparse", None)
        issues = []
        if dense:
            vs = getattr(dense, "vector_store", None)
            if vs and getattr(vs, "index", None) is None:
                issues.append("FAISS index not loaded")
        if sparse:
            bm25 = getattr(sparse, "bm25_store", None)
            if bm25 and getattr(bm25, "bm25", None) is None:
                issues.append("BM25 index not loaded")
        if issues:
            return ComponentStatus("index", False, "; ".join(issues))
        return ComponentStatus("index", True, "dense + sparse loaded")
