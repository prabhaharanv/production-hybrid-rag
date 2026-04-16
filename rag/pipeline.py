import re
from rag.prompting import build_rag_prompt, ABSTENTION_PHRASE


class RAGPipeline:
    def __init__(self, retriever, generator, reranker=None, query_rewriter=None):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.query_rewriter = query_rewriter

    def ask(self, question: str, top_k: int = 5) -> dict:
        # Step 1: Query rewriting
        rewritten_query = question
        if self.query_rewriter:
            rewritten_query = self.query_rewriter.rewrite(question)

        # Step 2: Hybrid retrieval (fetch extra candidates for reranking)
        fetch_k = top_k * 3 if self.reranker else top_k
        retrieved_chunks = self.retriever.retrieve(rewritten_query, top_k=fetch_k)

        # Step 3: Rerank and trim to top_k
        if self.reranker and retrieved_chunks:
            retrieved_chunks = self.reranker.rerank(rewritten_query, retrieved_chunks, top_k=top_k)

        # Step 4: Generate answer
        prompt = build_rag_prompt(question, retrieved_chunks)
        answer = self.generator.generate(prompt)

        # Step 5: Abstention check
        abstained = ABSTENTION_PHRASE in answer
        if abstained:
            answer = "I don't have enough information in the available documents to answer this question."

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

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "abstained": abstained,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }