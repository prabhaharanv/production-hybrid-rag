from rag.prompting import build_rag_prompt


class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, top_k: int = 5) -> dict:
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        prompt = build_rag_prompt(question, retrieved_chunks)
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
        }