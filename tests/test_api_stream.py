"""Tests for the /ask/stream SSE endpoint."""

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api as api_module


def _make_mock_pipeline():
    """Create a mock pipeline that yields SSE events."""
    mock = MagicMock()

    def fake_ask_stream(question, top_k=5):
        metadata = {
            "event": "metadata",
            "rewritten_query": question,
            "retrieved_chunks": [{"chunk_id": "c1", "doc_id": "d1", "title": "test.txt", "source": "test.txt", "score": 0.9}],
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        for token in ["Hello", " ", "world"]:
            yield f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"
        yield f"data: {json.dumps({'event': 'done', 'abstained': False, 'citations': ['test.txt']})}\n\n"

    mock.ask_stream = fake_ask_stream
    mock.ask.return_value = {
        "answer": "Hello world",
        "abstained": False,
        "citations": ["test.txt"],
        "chunks": [],
    }
    return mock


@pytest.fixture
def client():
    """Create a test client, then override the pipeline with a mock after lifespan runs."""
    with TestClient(api_module.app) as c:
        # Override the pipeline AFTER lifespan sets it
        original_pipeline = api_module.pipeline
        api_module.pipeline = _make_mock_pipeline()
        yield c
        api_module.pipeline = original_pipeline


class TestAskStream:
    def test_returns_200(self, client):
        resp = client.post("/ask/stream", json={"question": "What is RAG?"})
        assert resp.status_code == 200

    def test_content_type_is_sse(self, client):
        resp = client.post("/ask/stream", json={"question": "What is RAG?"})
        assert "text/event-stream" in resp.headers["content-type"]

    def test_emits_metadata_event_first(self, client):
        resp = client.post("/ask/stream", json={"question": "What is RAG?"})
        lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        assert len(lines) >= 1
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["event"] == "metadata"
        assert "rewritten_query" in first
        assert "retrieved_chunks" in first

    def test_emits_token_events(self, client):
        resp = client.post("/ask/stream", json={"question": "What is RAG?"})
        lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        token_events = [json.loads(l.removeprefix("data: ")) for l in lines if '"event": "token"' in l]
        assert len(token_events) == 3
        assert token_events[0]["data"] == "Hello"
        assert token_events[1]["data"] == " "
        assert token_events[2]["data"] == "world"

    def test_emits_done_event_last(self, client):
        resp = client.post("/ask/stream", json={"question": "What is RAG?"})
        lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        last = json.loads(lines[-1].removeprefix("data: "))
        assert last["event"] == "done"
        assert last["abstained"] is False
        assert "citations" in last

    def test_full_text_reconstructed(self, client):
        resp = client.post("/ask/stream", json={"question": "What is RAG?"})
        lines = [l for l in resp.text.split("\n") if l.startswith("data:")]
        tokens = []
        for line in lines:
            evt = json.loads(line.removeprefix("data: "))
            if evt["event"] == "token":
                tokens.append(evt["data"])
        assert "".join(tokens) == "Hello world"

    def test_requires_question_field(self, client):
        resp = client.post("/ask/stream", json={})
        assert resp.status_code == 422

    def test_respects_api_key_when_set(self):
        with TestClient(api_module.app) as c:
            api_module.pipeline = _make_mock_pipeline()
            original = api_module.settings.rag_api_key
            api_module.settings.rag_api_key = "secret-key"
            try:
                resp = c.post("/ask/stream", json={"question": "test"})
                # Without key should fail
                assert resp.status_code == 401
            finally:
                api_module.settings.rag_api_key = original

    def test_pipeline_none_returns_500(self):
        with TestClient(api_module.app) as c:
            original = api_module.pipeline
            api_module.pipeline = None
            try:
                resp = c.post("/ask/stream", json={"question": "test"})
                assert resp.status_code == 500
            finally:
                api_module.pipeline = original
