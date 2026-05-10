import json
import re
from collections.abc import Generator

from rag.prompting import build_rag_prompt, ABSTENTION_PHRASE
from app.observability.tracing import trace_span
from app.observability.metrics import get_metrics, track_step
from app.observability.logging import get_logger


class RAGPipeline:
    def __init__(
        self,
        retriever,
        generator,
        reranker=None,
        query_rewriter=None,
        cache=None,
        guardrails=None,
        compressor=None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.cache = cache
        self.guardrails = guardrails
        self.compressor = compressor
        self.log = get_logger("rag.pipeline")

    def ask(self, question: str, top_k: int = 5) -> dict:
        metrics = get_metrics()
        log = self.log.bind(question_len=len(question), top_k=top_k)

        # Step 0: Input guardrails
        if self.guardrails:
            guard_result = self.guardrails.check_input(question)
            if not guard_result.passed:
                log.warn("guardrail_blocked", violations=guard_result.violations)
                return {
                    "question": question,
                    "rewritten_query": question,
                    "answer": "I'm unable to process this query due to content policy.",
                    "abstained": True,
                    "citations": [],
                    "retrieved_chunks": [],
                    "guardrail_violations": guard_result.violations,
                }
            # Use redacted text if PII was found
            if guard_result.redacted_text:
                question = guard_result.redacted_text

        # Step 0b: Semantic cache lookup
        if self.cache:
            cached = self.cache.get(question)
            if cached is not None:
                log.info("cache_hit")
                return cached

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

        # Step 3b: Contextual compression
        if self.compressor and retrieved_chunks:
            with trace_span("compress", {"chunk_count": len(retrieved_chunks)}):
                with track_step("compress"):
                    retrieved_chunks = self.compressor.compress(rewritten_query, retrieved_chunks)

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

        result = {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "abstained": abstained,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }

        # Output guardrails
        if self.guardrails and not abstained:
            output_check = self.guardrails.check_output(answer)
            if not output_check.passed:
                log.warn("output_guardrail_triggered", violations=output_check.violations)
                result["answer"] = "The generated response was filtered due to content policy."
                result["abstained"] = True
                result["guardrail_violations"] = output_check.violations

        # Cache the result
        if self.cache and not result.get("abstained"):
            self.cache.put(question, result)

        return result

    def ask_stream(self, question: str, top_k: int = 5) -> Generator[str, None, None]:
        """Stream the RAG response as SSE events.

        Yields JSON-encoded SSE data lines:
        - {"event": "metadata", ...} with retrieval results
        - {"event": "token", "data": "..."} per token
        - {"event": "done", ...} with final metadata
        """
        metrics = get_metrics()
        log = self.log.bind(question_len=len(question), top_k=top_k, stream=True)

        # Input guardrails
        if self.guardrails:
            guard_result = self.guardrails.check_input(question)
            if not guard_result.passed:
                log.warn("guardrail_blocked_stream", violations=guard_result.violations)
                error_payload = {
                    "event": "error",
                    "message": "Query blocked by content policy.",
                    "violations": guard_result.violations,
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                return
            if guard_result.redacted_text:
                question = guard_result.redacted_text

        # Query rewriting
        rewritten_query = question
        if self.query_rewriter:
            with trace_span("rewrite", {"question_len": len(question)}):
                with track_step("rewrite"):
                    rewritten_query = self.query_rewriter.rewrite(question)

        # Hybrid retrieval
        fetch_k = top_k * 3 if self.reranker else top_k
        with trace_span("retrieve", {"fetch_k": fetch_k}):
            with track_step("retrieve"):
                retrieved_chunks = self.retriever.retrieve(rewritten_query, top_k=fetch_k)

        metrics.chunks_retrieved.observe(len(retrieved_chunks))

        # Rerank
        if self.reranker and retrieved_chunks:
            with trace_span("rerank", {"candidate_count": len(retrieved_chunks)}):
                with track_step("rerank"):
                    retrieved_chunks = self.reranker.rerank(rewritten_query, retrieved_chunks, top_k=top_k)

        # Contextual compression
        if self.compressor and retrieved_chunks:
            with trace_span("compress", {"chunk_count": len(retrieved_chunks)}):
                with track_step("compress"):
                    retrieved_chunks = self.compressor.compress(rewritten_query, retrieved_chunks)

        # Emit metadata event (sources, rewritten query)
        metadata = {
            "event": "metadata",
            "rewritten_query": rewritten_query,
            "retrieved_chunks": retrieved_chunks,
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        # Stream LLM tokens
        prompt = build_rag_prompt(question, retrieved_chunks)
        full_answer = []
        with trace_span("generate_stream", {"prompt_len": len(prompt)}):
            for token in self.generator.generate_stream(prompt):
                full_answer.append(token)
                yield f"data: {json.dumps({'event': 'token', 'data': token})}\n\n"

        answer = "".join(full_answer).strip()

        # Abstention check
        abstained = ABSTENTION_PHRASE in answer
        if abstained:
            metrics.abstention_count.inc()

        # Citations
        cited_indices = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer)))
        citations = []
        for idx in cited_indices:
            if 1 <= idx <= len(retrieved_chunks):
                chunk = retrieved_chunks[idx - 1]
                citations.append({"reference": idx, "title": chunk.get("title", ""), "source": chunk.get("source", "")})

        # Done event
        done_payload = {
            "event": "done",
            "abstained": abstained,
            "citations": citations,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

        log.info("stream_completed", abstained=abstained, citation_count=len(citations))