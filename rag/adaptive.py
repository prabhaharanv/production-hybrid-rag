"""Adaptive retrieval: routes queries to the most efficient retrieval strategy.

Simple keyword queries → BM25-only (fast, cheap)
Complex semantic queries → full hybrid retrieval (dense + sparse + rerank)
Factoid queries → dense-only (semantic similarity)

This reduces latency and compute cost for simple queries while preserving
quality for complex ones.
"""

from __future__ import annotations

import re

from openai import OpenAI


class QueryComplexityClassifier:
    """Classifies query complexity to route to appropriate retrieval strategy.

    Classification levels:
    - simple: keyword-like, factoid (e.g., "what is RAG?")
    - moderate: multi-part or comparative (e.g., "compare dense and sparse retrieval")
    - complex: abstract, multi-hop reasoning (e.g., "how does chunking affect RAG quality?")
    """

    # Heuristic signals for complexity
    _COMPLEX_INDICATORS = [
        re.compile(r"\b(compare|contrast|difference|versus|vs\.?)\b", re.IGNORECASE),
        re.compile(r"\b(how\s+does|why\s+does|what\s+happens\s+when)\b", re.IGNORECASE),
        re.compile(
            r"\b(trade-?off|pros?\s+and\s+cons?|advantage|disadvantage)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(explain|describe|elaborate)\s+.{20,}", re.IGNORECASE),
        re.compile(
            r"\b(relationship|correlation|impact|affect|influence)\b", re.IGNORECASE
        ),
    ]

    _SIMPLE_INDICATORS = [
        re.compile(r"^(what|who|where|when)\s+is\s+\w+", re.IGNORECASE),
        re.compile(r"^define\s+", re.IGNORECASE),
        re.compile(r"^\w+\s*\?$"),  # Single word question
    ]

    def classify(self, query: str) -> str:
        """Classify query complexity: 'simple', 'moderate', or 'complex'."""
        word_count = len(query.split())

        # Very short queries are simple
        if word_count <= 4:
            return "simple"

        # Check for complex indicators
        complex_score = sum(1 for p in self._COMPLEX_INDICATORS if p.search(query))
        simple_score = sum(1 for p in self._SIMPLE_INDICATORS if p.search(query))

        if complex_score >= 2 or (complex_score >= 1 and word_count > 15):
            return "complex"
        elif simple_score >= 1 or word_count <= 7:
            return "simple"
        else:
            return "moderate"


class LLMQueryClassifier:
    """Uses LLM to classify query complexity (higher accuracy, higher latency)."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def classify(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the complexity of this search query. "
                        "Respond with ONLY one word: 'simple', 'moderate', or 'complex'.\n"
                        "- simple: factoid, single-concept lookup\n"
                        "- moderate: multi-part or needs some reasoning\n"
                        "- complex: comparative, multi-hop, abstract reasoning"
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().lower()
        if result in ("simple", "moderate", "complex"):
            return result
        return "moderate"  # Default fallback


class AdaptiveRetriever:
    """Routes queries to the most efficient retrieval strategy based on complexity.

    Strategy routing:
    - simple → sparse-only (BM25) — fast keyword matching
    - moderate → dense-only — semantic embedding search
    - complex → full hybrid (dense + sparse + RRF) — maximum recall
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        hybrid_retriever,
        classifier: QueryComplexityClassifier | LLMQueryClassifier | None = None,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.hybrid = hybrid_retriever
        self.classifier = classifier or QueryComplexityClassifier()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Route query to appropriate retriever based on complexity."""
        complexity = self.classifier.classify(query)

        if complexity == "simple":
            results = self.sparse.retrieve(query, top_k=top_k)
            strategy_used = "sparse"
        elif complexity == "moderate":
            results = self.dense.retrieve(query, top_k=top_k)
            strategy_used = "dense"
        else:  # complex
            results = self.hybrid.retrieve(query, top_k=top_k)
            strategy_used = "hybrid"

        # Annotate results with routing info
        for r in results:
            r["retrieval_strategy"] = strategy_used
            r["query_complexity"] = complexity

        return results
