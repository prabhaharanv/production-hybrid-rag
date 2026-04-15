import re
from rag.prompting import build_rag_prompt


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, top_k: int = 5) -> dict:
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        prompt = build_rag_prompt(question, retrieved_chunks)
        answer = self.generator.generate(prompt)

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
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
        }