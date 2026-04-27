import re
import time

from rag.prompting import build_rag_prompt, ABSTENTION_PHRASE
from app.observability.tracing import trace_span
from app.observability.metrics import get_metrics, track_step
from app.observability.logging import get_logger


class RAGPipeline:
    def __init__(self, retriever, generator, reranker=None, query_rewriter=None):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.log = get_logger("rag.pipeline")

    def ask(self, question: str, top_k: int = 5) -> dict:
        metrics = get_metrics()
        log = self.log.bind(question_len=len(question), top_k=top_k)

        # Step 1: Query rewriting
        rewritten_query = question
        if self.query_rewriter:
            with trace_span("rewrite", {"question_len": len(question)}):
                with track_step("rewrite"):
                    rewritten_query = self.query_rewriter.rewrite(question)
            log = log.bind(rewritten=True)
        else:
            log = log.bind(rewritten=False)

        # Step 2: Hybrid retrieval (fetch extra candidates for reranking)
        fetch_k = top_k * 3 if self.reranker else top_k
        with trace_span("retrieve", {"fetch_k": fetch_k}) as span:
            with track_step("retrieve"):
                retrieved_chunks = self.retriever.retrieve(rewritten_query, top_k=fetch_k)
            span.set_attribute("chunk_count", len(retrieved_chunks))

        metrics.chunks_retrieved.observe(len(retrieved_chunks))
        if retrieved_chunks:
            top_score = max(c.get("score", 0) for c in retrieved_chunks)
            metrics.retrieval_score.observe(top_score)

        # Step 3: Rerank and trim to top_k
        if self.reranker and retrieved_chunks:
            with trace_span("rerank", {"candidate_count": len(retrieved_chunks)}):
                with track_step("rerank"):
                    retrieved_chunks = self.reranker.rerank(rewritten_query, retrieved_chunks, top_k=top_k)

        # Step 4: Generate answer
        prompt = build_rag_prompt(question, retrieved_chunks)
        with trace_span("generate", {"prompt_len": len(prompt)}) as span:
            with track_step("generate"):
                answer = self.generator.generate(prompt)
            span.set_attribute("answer_len", len(answer))

        # Approximate token usage (1 token ≈ 4 chars)
        metrics.token_usage.labels(type="prompt").inc(len(prompt) // 4)
        metrics.token_usage.labels(type="completion").inc(len(answer) // 4)

        # Step 5: Abstention check
        abstained = ABSTENTION_PHRASE in answer
        if abstained:
            answer = "I don't have enough information in the available documents to answer this question."
            metrics.abstention_count.inc()
            log.info("request_abstained")

        # Step 6: Extract citations
        cited_indices = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer)))
        citations = []
        for idx in cited_indices:
            if 1 <= idx <= len(retrieved_chunks):
                chunk = retrieved_chunks[idx - 1]
                citations.append(
                    {
                        "reference": idx,
                        "title": chunk.get("title", ""),
                        "source": chunk.get("source", ""),
                    }
                )

        log.info(
            "request_completed",
            abstained=abstained,
            citation_count=len(citations),
            chunk_count=len(retrieved_chunks),
        )

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "abstained": abstained,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }