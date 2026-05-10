"""
Locust load tests for the Hybrid RAG API.

Usage:
    # Web UI (default):
    locust -f loadtest/locustfile.py --host http://localhost:8000

    # Headless — quick smoke test:
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 -t 60s

    # Headless — sustained load:
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 100 -r 10 -t 5m

    # With API key:
    RAG_API_KEY=your-key locust -f loadtest/locustfile.py --host http://localhost:8000
"""

import os
import random

from locust import HttpUser, task, between, tag


# Sample questions spanning different topics and complexity levels
QUESTIONS = [
    "What is RAG?",
    "How does chunking work?",
    "What is reciprocal rank fusion?",
    "Explain the difference between dense and sparse retrieval",
    "How does cross-encoder reranking improve results?",
    "What is answer abstention?",
    "How are embeddings created from documents?",
    "What is BM25?",
    "How does FAISS store vectors?",
    "What is query rewriting?",
    "Explain hybrid search in RAG systems",
    "What is the purpose of overlap in text chunking?",
    "How does the generator produce answers?",
    "What are citation references in RAG?",
    "How do you evaluate a RAG system?",
]

# Out-of-scope questions — should trigger abstention
ABSTENTION_QUESTIONS = [
    "What is the capital of France?",
    "How do I cook pasta?",
    "What is quantum computing?",
    "Who won the world cup in 2022?",
]


class RAGUser(HttpUser):
    """Simulates a user interacting with the RAG API."""

    weight = 10  # 10:1 ratio vs HealthOnlyUser
    wait_time = between(1, 5)

    def on_start(self):
        """Set up auth header if RAG_API_KEY is provided."""
        api_key = os.getenv("RAG_API_KEY", "")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key

    # ── Primary endpoint: /ask ──

    @tag("ask", "core")
    @task(10)
    def ask_question(self):
        """POST /ask with a random in-scope question."""
        question = random.choice(QUESTIONS)
        with self.client.post(
            "/ask",
            json={"question": question},
            headers=self.headers,
            name="/ask",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Empty answer in response")
            elif resp.status_code == 429:
                resp.success()  # rate limiting is expected under load
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:200]}")

    @tag("ask", "core")
    @task(3)
    def ask_with_top_k(self):
        """POST /ask with a custom top_k value."""
        question = random.choice(QUESTIONS)
        top_k = random.choice([1, 3, 5, 10])
        with self.client.post(
            "/ask",
            json={"question": question, "top_k": top_k},
            headers=self.headers,
            name="/ask (top_k)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if len(data.get("retrieved_chunks", [])) > top_k:
                    resp.failure(f"Got {len(data['retrieved_chunks'])} chunks, expected <= {top_k}")
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @tag("ask", "abstention")
    @task(2)
    def ask_out_of_scope(self):
        """POST /ask with an out-of-scope question — should abstain."""
        question = random.choice(ABSTENTION_QUESTIONS)
        with self.client.post(
            "/ask",
            json={"question": question},
            headers=self.headers,
            name="/ask (abstention)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    # ── Health endpoints ──

    @tag("health")
    @task(5)
    def health_check(self):
        """GET /health — lightweight liveness probe."""
        self.client.get("/health", name="/health")

    @tag("health")
    @task(2)
    def readiness_check(self):
        """GET /health/ready — deep readiness probe."""
        with self.client.get(
            "/health/ready",
            name="/health/ready",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("healthy"):
                    resp.failure("Pipeline not healthy")
            else:
                resp.failure(f"Readiness returned {resp.status_code}")

    # ── Metrics endpoint ──

    @tag("metrics")
    @task(1)
    def scrape_metrics(self):
        """GET /metrics — Prometheus scrape endpoint."""
        with self.client.get(
            "/metrics",
            name="/metrics",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                if b"rag_requests_total" not in resp.content:
                    resp.failure("Missing expected metric rag_requests_total")
            else:
                resp.failure(f"Metrics returned {resp.status_code}")


class HealthOnlyUser(HttpUser):
    """Lightweight user that only hits health endpoints.
    Use to verify the service stays alive during heavy /ask load.
    Run with: locust -f locustfile.py --tags health
    """

    wait_time = between(2, 10)
    weight = 1  # spawn far fewer of these

    @tag("health")
    @task
    def health(self):
        self.client.get("/health", name="/health (monitor)")

    @tag("health")
    @task
    def ready(self):
        self.client.get("/health/ready", name="/health/ready (monitor)")
