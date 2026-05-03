# Evaluation Metrics for RAG Systems

Evaluating Retrieval-Augmented Generation systems requires metrics that capture both retrieval quality and generation faithfulness.

## Retrieval Metrics

### Mean Reciprocal Rank (MRR)

MRR measures how high the first relevant document appears in the ranked list. It is computed as the average of 1/rank across all queries. A perfect MRR of 1.0 means the correct document is always ranked first.

### Normalized Discounted Cumulative Gain (NDCG@K)

NDCG accounts for the position of relevant documents in the ranking. Documents ranked higher contribute more to the score. It normalizes the Discounted Cumulative Gain by the ideal ranking, producing a score between 0 and 1. NDCG@5 is commonly used for RAG evaluation since most systems retrieve 3-5 passages.

### Recall@K

Recall@K measures the fraction of relevant documents that appear in the top-K retrieved results. High recall is essential for RAG because missing a relevant passage means the generator cannot use that information.

## Generation Metrics

### RAGAS Faithfulness

Faithfulness measures whether the generated answer is grounded in the retrieved context. It decomposes the answer into individual claims and checks each claim against the context. The score is the ratio of supported claims to total claims. A faithfulness score below 0.8 indicates potential hallucination.

### RAGAS Answer Relevance

Answer relevance measures whether the response actually addresses the question asked. It generates synthetic questions from the answer and computes cosine similarity between their embeddings and the original question embedding. Low relevance indicates the model went off-topic.

### BERTScore

BERTScore uses contextual embeddings from BERT to compute token-level similarity between the generated answer and a reference answer. Unlike BLEU or ROUGE which rely on exact n-gram matches, BERTScore captures semantic similarity. It reports precision, recall, and F1 scores.

### Context Precision and Context Recall

Context precision measures what fraction of the retrieved passages are actually relevant to answering the question. Context recall measures what fraction of the information needed to answer the question is present in the retrieved context. Together they indicate whether the retriever fetches the right passages without too much noise.

## Hallucination Detection

Natural Language Inference (NLI) models classify each claim in the generated answer as entailed, contradicted, or neutral with respect to the context. Claims classified as contradicted are hallucinations. The hallucination rate is the fraction of claims that are contradicted or neutral (unsupported).

Cross-encoder models like DeBERTa-v3 trained on NLI datasets are commonly used. They process (context, claim) pairs and output probabilities for each class.

## End-to-End Metrics

### Abstention Accuracy

A well-calibrated RAG system should refuse to answer when the context is insufficient. Abstention accuracy measures whether the system correctly identifies out-of-scope questions. False answers to out-of-scope questions are particularly harmful as they present fabricated information with apparent confidence.

### Latency Breakdown

Production RAG systems track latency at each pipeline stage: retrieval (typically 10-50ms), reranking (50-200ms), and generation (500-5000ms). P95 and P99 latency are more informative than averages since LLM generation time has high variance.
