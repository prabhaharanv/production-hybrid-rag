"""
Mathematical Evaluation Framework for RAG systems.

Metrics implemented:
- RAGAS Faithfulness
- RAGAS Answer Relevance
- Context Precision @ K
- Context Recall
- BERTScore (F1)
- MRR (Mean Reciprocal Rank)
- NDCG @ K (Normalized Discounted Cumulative Gain)
- Hallucination Detection (NLI-based)
"""

import math
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ---------------------------------------------------------------------------
# Lazy model singletons (loaded once on first use)
# ---------------------------------------------------------------------------

_nli_model: CrossEncoder | None = None
_embed_model: SentenceTransformer | None = None
_bertscore_model: SentenceTransformer | None = None

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
BERTSCORE_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(NLI_MODEL_NAME)
    return _nli_model


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_bertscore_model() -> SentenceTransformer:
    global _bertscore_model
    if _bertscore_model is None:
        _bertscore_model = SentenceTransformer(BERTSCORE_MODEL_NAME)
    return _bertscore_model


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of A and rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


# ---------------------------------------------------------------------------
# NLI labels: 0 = contradiction, 1 = entailment, 2 = neutral
# (order depends on model; deberta-v3-small uses this mapping)
# ---------------------------------------------------------------------------

_NLI_ENTAILMENT = 1
_NLI_CONTRADICTION = 0
_NLI_NEUTRAL = 2


# ---------------------------------------------------------------------------
# RAGAS Faithfulness
# ---------------------------------------------------------------------------


def ragas_faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness = (claims in answer supported by context) / (total claims)

    Each sentence in the answer is treated as a claim.
    A claim is *supported* if the NLI model classifies (context, claim) as entailment.
    """
    claims = _split_sentences(answer)
    if not claims:
        return 1.0

    model = _get_nli_model()
    pairs = [(context, claim) for claim in claims]
    scores = model.predict(pairs)

    supported = sum(1 for s in scores if np.argmax(s) == _NLI_ENTAILMENT)
    return supported / len(claims)


# ---------------------------------------------------------------------------
# RAGAS Answer Relevance
# ---------------------------------------------------------------------------


def ragas_answer_relevance(question: str, answer: str, n_reverse: int = 3) -> float:
    """
    Relevance = (1/N) * Σ cos(E(q), E(a_i))

    where a_i are *reverse-generated* questions.  Because we don't call an LLM
    here, we approximate reverse questions by treating each answer sentence as
    a pseudo-question.  For a full implementation, plug in an LLM to generate
    questions from the answer.
    """
    model = _get_embed_model()
    answer_sentences = _split_sentences(answer)
    if not answer_sentences:
        return 0.0

    # Use up to n_reverse answer sentences as proxy reverse questions
    reverse_qs = answer_sentences[:n_reverse]

    q_emb = model.encode([question])
    a_embs = model.encode(reverse_qs)

    similarities = [_cosine_similarity(q_emb[0], a_emb) for a_emb in a_embs]
    return float(np.mean(similarities))


# ---------------------------------------------------------------------------
# Context Precision @ K
# ---------------------------------------------------------------------------


def context_precision_at_k(
    retrieved_chunks: list[dict],
    relevant_source: str | None,
    k: int | None = None,
) -> float:
    """
    CP@K = (1/K) * Σ_{k=1}^{K} Precision(k) * rel(k)

    rel(k) = 1 if chunk k is from the expected source, else 0.
    Precision(k) = (# relevant in top-k) / k
    """
    if relevant_source is None:
        return 1.0

    chunks = retrieved_chunks[:k] if k else retrieved_chunks
    K = len(chunks)
    if K == 0:
        return 0.0

    relevance = []
    for chunk in chunks:
        title = chunk.get("title", "")
        source = chunk.get("source", "")
        rel = 1.0 if (relevant_source in title or relevant_source in source) else 0.0
        relevance.append(rel)

    weighted_sum = 0.0
    running_relevant = 0.0
    for i, rel in enumerate(relevance):
        running_relevant += rel
        precision_at_i = running_relevant / (i + 1)
        weighted_sum += precision_at_i * rel

    return weighted_sum / K


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------


def context_recall(ground_truth: str, context: str) -> float:
    """
    CR = (ground truth sentences attributable to context) / (total ground truth sentences)

    A ground-truth sentence is *attributable* if the NLI model classifies
    (context, gt_sentence) as entailment.
    """
    gt_sentences = _split_sentences(ground_truth)
    if not gt_sentences:
        return 1.0

    model = _get_nli_model()
    pairs = [(context, gt) for gt in gt_sentences]
    scores = model.predict(pairs)

    attributable = sum(1 for s in scores if np.argmax(s) == _NLI_ENTAILMENT)
    return attributable / len(gt_sentences)


# ---------------------------------------------------------------------------
# BERTScore (F1)
# ---------------------------------------------------------------------------


def bert_score_f1(reference: str, candidate: str) -> dict[str, float]:
    """
    F_BERT = 2 * (P_BERT * R_BERT) / (P_BERT + R_BERT)

    Using contextual embeddings from a sentence-transformer.
    - P_BERT: for each candidate token-sentence, max similarity to any reference token-sentence
    - R_BERT: for each reference token-sentence, max similarity to any candidate token-sentence

    We operate at sentence granularity (sentence-level BERTScore).
    """
    ref_sents = _split_sentences(reference)
    cand_sents = _split_sentences(candidate)

    if not ref_sents or not cand_sents:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = _get_bertscore_model()
    ref_embs = model.encode(ref_sents)
    cand_embs = model.encode(cand_sents)

    sim_matrix = _cosine_similarity_matrix(cand_embs, ref_embs)  # (C, R)

    # Precision: for each candidate sentence, best match to any reference
    p_bert = float(np.mean(np.max(sim_matrix, axis=1)))
    # Recall: for each reference sentence, best match to any candidate
    r_bert = float(np.mean(np.max(sim_matrix, axis=0)))

    if p_bert + r_bert == 0:
        f1 = 0.0
    else:
        f1 = 2 * (p_bert * r_bert) / (p_bert + r_bert)

    return {
        "precision": round(p_bert, 4),
        "recall": round(r_bert, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# MRR & NDCG @ K
# ---------------------------------------------------------------------------


def mean_reciprocal_rank(
    retrieved_chunks: list[dict],
    relevant_source: str | None,
) -> float:
    """
    MRR = 1 / rank_of_first_relevant_result

    Returns 0 if no relevant result is found.
    """
    if relevant_source is None:
        return 1.0

    for i, chunk in enumerate(retrieved_chunks):
        title = chunk.get("title", "")
        source = chunk.get("source", "")
        if relevant_source in title or relevant_source in source:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_chunks: list[dict],
    relevant_source: str | None,
    k: int | None = None,
) -> float:
    """
    NDCG@K = DCG@K / IDCG@K

    DCG@K  = Σ_{i=1}^{K} rel(i) / log2(i + 1)
    IDCG@K = ideal DCG (all relevant docs at the top)
    """
    if relevant_source is None:
        return 1.0

    chunks = retrieved_chunks[:k] if k else retrieved_chunks
    K = len(chunks)
    if K == 0:
        return 0.0

    rels = []
    for chunk in chunks:
        title = chunk.get("title", "")
        source = chunk.get("source", "")
        rel = 1.0 if (relevant_source in title or relevant_source in source) else 0.0
        rels.append(rel)

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    # IDCG (sort rels descending = ideal ranking)
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Hallucination Detection (NLI-based)
# ---------------------------------------------------------------------------


def hallucination_score(answer: str, context: str) -> dict:
    """
    Classify each answer claim as entailed / contradicted / neutral vs context.

    Returns:
        {
            "claims": [...],
            "labels": [...],          # "entailed" | "contradicted" | "neutral"
            "hallucination_rate": float  # fraction that are contradicted or neutral
        }
    """
    claims = _split_sentences(answer)
    if not claims:
        return {"claims": [], "labels": [], "hallucination_rate": 0.0}

    model = _get_nli_model()
    pairs = [(context, claim) for claim in claims]
    scores = model.predict(pairs)

    label_map = {
        _NLI_CONTRADICTION: "contradicted",
        _NLI_ENTAILMENT: "entailed",
        _NLI_NEUTRAL: "neutral",
    }

    labels = [label_map[int(np.argmax(s))] for s in scores]
    hallucinated = sum(1 for label in labels if label != "entailed")
    rate = hallucinated / len(claims)

    return {
        "claims": claims,
        "labels": labels,
        "hallucination_rate": round(rate, 4),
    }


# ---------------------------------------------------------------------------
# Convenience: run all metrics on a single evaluation item
# ---------------------------------------------------------------------------


def evaluate_single(
    question: str,
    answer: str,
    context: str,
    retrieved_chunks: list[dict],
    ground_truth: str | None = None,
    relevant_source: str | None = None,
    k: int | None = None,
) -> dict:
    """Compute all metrics for one question-answer pair."""
    ctx = context or ""

    bs = bert_score_f1(ground_truth or question, answer)
    hall = hallucination_score(answer, ctx)

    result = {
        "faithfulness": round(ragas_faithfulness(answer, ctx), 4),
        "answer_relevance": round(ragas_answer_relevance(question, answer), 4),
        "context_precision": round(
            context_precision_at_k(retrieved_chunks, relevant_source, k), 4
        ),
        "context_recall": round(context_recall(ground_truth or "", ctx), 4)
        if ground_truth
        else None,
        "bertscore_precision": bs["precision"],
        "bertscore_recall": bs["recall"],
        "bertscore_f1": bs["f1"],
        "mrr": round(mean_reciprocal_rank(retrieved_chunks, relevant_source), 4),
        "ndcg": round(ndcg_at_k(retrieved_chunks, relevant_source, k), 4),
        "hallucination_rate": hall["hallucination_rate"],
        "hallucination_details": hall,
    }
    return result
