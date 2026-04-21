import math
import pytest
import numpy as np

from eval.metrics import (
    _split_sentences,
    _cosine_similarity,
    context_precision_at_k,
    context_recall,
    mean_reciprocal_rank,
    ndcg_at_k,
    ragas_faithfulness,
    ragas_answer_relevance,
    bert_score_f1,
    hallucination_score,
    evaluate_single,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_basic(self):
        assert _split_sentences("Hello world. How are you?") == ["Hello world.", "How are you?"]

    def test_single_sentence(self):
        assert _split_sentences("Hello world.") == ["Hello world."]

    def test_empty(self):
        assert _split_sentences("") == []

    def test_no_punctuation(self):
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]


class TestCosineSimilarity:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0


# ── Context Precision @ K ────────────────────────────────────────────────────

class TestContextPrecision:
    def test_all_relevant(self):
        chunks = [{"source": "data/raw/a.txt"}, {"source": "data/raw/a.txt"}]
        assert context_precision_at_k(chunks, "a.txt") == pytest.approx(1.0)

    def test_none_relevant(self):
        chunks = [{"source": "data/raw/b.txt"}, {"source": "data/raw/c.txt"}]
        assert context_precision_at_k(chunks, "a.txt") == pytest.approx(0.0)

    def test_first_relevant(self):
        chunks = [{"source": "data/raw/a.txt"}, {"source": "data/raw/b.txt"}]
        # rel = [1, 0], precision(1)=1*1=1, precision(2)=0.5*0=0 => (1+0)/2 = 0.5
        assert context_precision_at_k(chunks, "a.txt") == pytest.approx(0.5)

    def test_none_expected(self):
        assert context_precision_at_k([], None) == 1.0

    def test_empty_chunks(self):
        assert context_precision_at_k([], "a.txt") == 0.0

    def test_k_limits(self):
        chunks = [{"source": "a.txt"}, {"source": "b.txt"}, {"source": "a.txt"}]
        result = context_precision_at_k(chunks, "a.txt", k=1)
        assert result == pytest.approx(1.0)


# ── MRR ──────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_position(self):
        chunks = [{"title": "a.txt"}, {"title": "b.txt"}]
        assert mean_reciprocal_rank(chunks, "a.txt") == pytest.approx(1.0)

    def test_second_position(self):
        chunks = [{"title": "b.txt"}, {"title": "a.txt"}]
        assert mean_reciprocal_rank(chunks, "a.txt") == pytest.approx(0.5)

    def test_not_found(self):
        chunks = [{"title": "b.txt"}, {"title": "c.txt"}]
        assert mean_reciprocal_rank(chunks, "a.txt") == 0.0

    def test_none_expected(self):
        assert mean_reciprocal_rank([], None) == 1.0


# ── NDCG @ K ────────────────────────────────────────────────────────────────

class TestNDCG:
    def test_perfect_ranking(self):
        chunks = [{"source": "a.txt"}, {"source": "a.txt"}, {"source": "b.txt"}]
        assert ndcg_at_k(chunks, "a.txt") == pytest.approx(1.0)

    def test_no_relevant(self):
        chunks = [{"source": "b.txt"}, {"source": "c.txt"}]
        assert ndcg_at_k(chunks, "a.txt") == pytest.approx(0.0)

    def test_none_expected(self):
        assert ndcg_at_k([], None) == 1.0

    def test_reversed_ranking(self):
        # Relevant at position 2, irrelevant at position 1
        chunks = [{"source": "b.txt"}, {"source": "a.txt"}]
        # DCG = 0/log2(2) + 1/log2(3) = 0 + 0.6309
        # IDCG = 1/log2(2) + 0/log2(3) = 1.0
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(chunks, "a.txt") == pytest.approx(expected, rel=1e-3)

    def test_k_limit(self):
        chunks = [{"source": "b.txt"}, {"source": "b.txt"}, {"source": "a.txt"}]
        # With k=2, only first two (irrelevant) are considered
        assert ndcg_at_k(chunks, "a.txt", k=2) == pytest.approx(0.0)


# ── NLI-based metrics (faithfulness, context recall, hallucination) ──────────
# These require model loading so we mark them for optional slow runs.

@pytest.mark.slow
class TestRagasFaithfulness:
    def test_supported_claim(self):
        context = "Paris is the capital of France. It is located in Europe."
        answer = "Paris is the capital of France."
        score = ragas_faithfulness(answer, context)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_empty_answer(self):
        assert ragas_faithfulness("", "Some context.") == 1.0


@pytest.mark.slow
class TestContextRecall:
    def test_attributable(self):
        context = "The cat sat on the mat. It was a sunny day."
        ground_truth = "The cat sat on the mat."
        score = context_recall(ground_truth, context)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_empty_ground_truth(self):
        assert context_recall("", "Some context.") == 1.0


@pytest.mark.slow
class TestAnswerRelevance:
    def test_relevant_answer(self):
        question = "What is the capital of France?"
        answer = "The capital of France is Paris. Paris is a major European city."
        score = ragas_answer_relevance(question, answer)
        assert 0.0 <= score <= 1.0

    def test_empty_answer(self):
        assert ragas_answer_relevance("Question?", "") == 0.0


@pytest.mark.slow
class TestBertScore:
    def test_identical(self):
        text = "The cat sat on the mat."
        result = bert_score_f1(text, text)
        assert result["f1"] > 0.9

    def test_different(self):
        ref = "The cat sat on the mat."
        cand = "Quantum mechanics describes particle behavior."
        result = bert_score_f1(ref, cand)
        assert result["f1"] < result["precision"] or result["f1"] < 0.9

    def test_empty(self):
        result = bert_score_f1("", "something")
        assert result["f1"] == 0.0


@pytest.mark.slow
class TestHallucinationScore:
    def test_grounded_answer(self):
        context = "Python is a programming language created by Guido van Rossum."
        answer = "Python is a programming language."
        result = hallucination_score(answer, context)
        assert result["hallucination_rate"] <= 0.5

    def test_empty_answer(self):
        result = hallucination_score("", "Context.")
        assert result["hallucination_rate"] == 0.0
        assert result["claims"] == []


@pytest.mark.slow
class TestEvaluateSingle:
    def test_returns_all_keys(self):
        result = evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval-augmented generation.",
            context="RAG stands for retrieval-augmented generation. It combines retrieval with generation.",
            retrieved_chunks=[{"title": "rag.txt", "source": "data/raw/rag.txt", "text": "RAG stands for..."}],
            ground_truth="RAG is retrieval-augmented generation.",
            relevant_source="rag.txt",
            k=5,
        )
        expected_keys = {
            "faithfulness", "answer_relevance", "context_precision",
            "context_recall", "bertscore_precision", "bertscore_recall",
            "bertscore_f1", "mrr", "ndcg", "hallucination_rate",
            "hallucination_details",
        }
        assert expected_keys.issubset(result.keys())
        for key in expected_keys - {"hallucination_details", "context_recall"}:
            assert isinstance(result[key], float)
