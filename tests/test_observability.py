import pytest
import time
from prometheus_client import CollectorRegistry

from app.observability.tracing import init_tracing, get_tracer, trace_span
from app.observability.metrics import RAGMetrics, init_metrics, get_metrics, track_request, track_step
from app.observability.logging import init_logging, get_logger, new_correlation_id
from app.observability.health import HealthChecker


# ---- Tracing Tests ----

class TestTracing:
    def test_init_tracing_returns_tracer(self):
        tracer = init_tracing(service_name="test-service")
        assert tracer is not None

    def test_get_tracer_returns_instance(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_trace_span_context_manager(self):
        init_tracing(service_name="test")
        with trace_span("test-span", {"key": "value"}) as span:
            assert span is not None

    def test_trace_span_records_exception(self):
        init_tracing(service_name="test")
        with pytest.raises(ValueError):
            with trace_span("failing-span"):
                raise ValueError("test error")


# ---- Metrics Tests ----

class TestMetrics:
    def _fresh_metrics(self):
        return RAGMetrics(registry=CollectorRegistry())

    def test_request_count_increments(self):
        m = self._fresh_metrics()
        m.request_count.labels(status="success").inc()
        assert m.request_count.labels(status="success")._value.get() == 1.0

    def test_request_latency_observes(self):
        m = self._fresh_metrics()
        m.request_latency.observe(0.5)
        assert m.request_latency._sum.get() == 0.5

    def test_step_latency_by_label(self):
        m = self._fresh_metrics()
        m.step_latency.labels(step="retrieve").observe(0.1)
        m.step_latency.labels(step="generate").observe(0.3)
        assert m.step_latency.labels(step="retrieve")._sum.get() == 0.1
        assert m.step_latency.labels(step="generate")._sum.get() == 0.3

    def test_abstention_count(self):
        m = self._fresh_metrics()
        m.abstention_count.inc()
        m.abstention_count.inc()
        assert m.abstention_count._value.get() == 2.0

    def test_token_usage_labels(self):
        m = self._fresh_metrics()
        m.token_usage.labels(type="prompt").inc(100)
        m.token_usage.labels(type="completion").inc(50)
        assert m.token_usage.labels(type="prompt")._value.get() == 100.0
        assert m.token_usage.labels(type="completion")._value.get() == 50.0

    def test_pipeline_ready_gauge(self):
        m = self._fresh_metrics()
        m.pipeline_ready.set(1)
        assert m.pipeline_ready._value.get() == 1.0
        m.pipeline_ready.set(0)
        assert m.pipeline_ready._value.get() == 0.0

    def test_generate_latest_returns_bytes(self):
        m = self._fresh_metrics()
        output = m.generate_latest()
        assert isinstance(output, bytes)
        assert b"rag_requests_total" in output

    def test_track_request_context_manager(self):
        registry = CollectorRegistry()
        init_metrics(registry=registry)
        with track_request() as m:
            pass  # simulate successful request
        assert m.request_count.labels(status="success")._value.get() == 1.0

    def test_track_request_records_error(self):
        registry = CollectorRegistry()
        init_metrics(registry=registry)
        with pytest.raises(RuntimeError):
            with track_request():
                raise RuntimeError("boom")
        m = get_metrics()
        assert m.request_count.labels(status="error")._value.get() == 1.0

    def test_track_step_records_latency(self):
        registry = CollectorRegistry()
        init_metrics(registry=registry)
        with track_step("retrieve"):
            time.sleep(0.01)
        m = get_metrics()
        assert m.step_latency.labels(step="retrieve")._sum.get() > 0


# ---- Logging Tests ----

class TestLogging:
    def test_init_logging_no_crash(self):
        init_logging(log_level="DEBUG", json_output=True)

    def test_init_logging_console_mode(self):
        init_logging(log_level="INFO", json_output=False)

    def test_get_logger_returns_bound_logger(self):
        init_logging()
        log = get_logger("test")
        assert log is not None

    def test_new_correlation_id_format(self):
        cid = new_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 16

    def test_correlation_ids_are_unique(self):
        ids = {new_correlation_id() for _ in range(100)}
        assert len(ids) == 100


# ---- Health Check Tests ----

class TestHealthChecker:
    def test_liveness_always_ok(self):
        hc = HealthChecker()
        assert hc.liveness() == {"status": "ok"}

    def test_readiness_no_pipeline(self):
        hc = HealthChecker()
        result = hc.readiness()
        assert result["healthy"] is False
        assert result["status"] == "degraded"

    def test_readiness_with_pipeline(self):
        class FakeRetriever:
            def retrieve(self, query, top_k=1):
                return [{"text": "probe", "score": 0.5}]

        class FakeGenerator:
            client = object()  # truthy client

        class FakePipeline:
            retriever = FakeRetriever()
            generator = FakeGenerator()

        hc = HealthChecker()
        hc.set_pipeline(FakePipeline())
        result = hc.readiness()
        assert result["healthy"] is True
        assert result["status"] == "ready"

    def test_readiness_retriever_failure(self):
        class BrokenRetriever:
            def retrieve(self, query, top_k=1):
                raise ConnectionError("DB down")

        class FakeGenerator:
            client = object()

        class FakePipeline:
            retriever = BrokenRetriever()
            generator = FakeGenerator()

        hc = HealthChecker()
        hc.set_pipeline(FakePipeline())
        result = hc.readiness()
        assert result["healthy"] is False
        retriever_status = next(c for c in result["components"] if c["name"] == "retriever")
        assert retriever_status["healthy"] is False
        assert "DB down" in retriever_status["detail"]

    def test_readiness_generator_no_client(self):
        class FakeRetriever:
            def retrieve(self, query, top_k=1):
                return []

        class FakeGenerator:
            client = None

        class FakePipeline:
            retriever = FakeRetriever()
            generator = FakeGenerator()

        hc = HealthChecker()
        hc.set_pipeline(FakePipeline())
        result = hc.readiness()
        assert result["healthy"] is False

    def test_set_pipeline_to_none(self):
        hc = HealthChecker()
        hc.set_pipeline(object())
        hc.set_pipeline(None)
        assert hc.pipeline is None
        assert hc.readiness()["healthy"] is False
