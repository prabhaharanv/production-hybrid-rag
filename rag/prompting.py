def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"[Chunk {i}]\n"
            f"Title: {chunk['title']}\n"
            f"Source: {chunk['source']}\n"
            f"Text: {chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"""You are a helpful assistant answering questions using only the retrieved context.

Rules:
- Use only the context below.
- If the answer is not supported by the context, say that you do not have enough information.
- Be concise but accurate.
- When relevant, mention the source title.

Question:
{question}

Retrieved Context:
{context}

Answer:
"""