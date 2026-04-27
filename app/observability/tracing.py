"""
Distributed Tracing with OpenTelemetry.

Traces each pipeline step: rewrite → retrieve → rerank → generate
with latency per step. Exports spans via OTLP to a collector (Jaeger/Tempo).
"""

import time
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

_tracer: trace.Tracer | None = None


def init_tracing(
    service_name: str = "rag-api",
    otlp_endpoint: str | None = None,
) -> trace.Tracer:
    """Initialise the OpenTelemetry tracer provider.

    If *otlp_endpoint* is provided (e.g. ``http://otel-collector:4317``),
    spans are exported via OTLP/gRPC.  Otherwise they go to the console
    (useful for local development).
    """
    global _tracer

    resource = Resource.create({ResourceAttributes.SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Return the initialised tracer (falls back to noop if not yet init'd)."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("rag-api")
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict | None = None):
    """Context manager that creates a child span and records duration.

    Usage::

        with trace_span("retrieve", {"top_k": 5}) as span:
            chunks = retriever.retrieve(query)
            span.set_attribute("chunk_count", len(chunks))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.set_status(trace.StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("duration_ms", round(duration_ms, 2))
