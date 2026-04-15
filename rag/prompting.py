def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_label = chunk.get("title", "Unknown")
        block = (
            f"[{i}] \"{source_label}\"\n"
            f"    Source: {chunk.get('source', 'N/A')}\n"
            f"    ---\n"
            f"    {chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    source_list = ", ".join(
        f"[{i}] {chunk.get('title', 'Unknown')}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    )

    return f"""You are a helpful assistant answering questions using only the retrieved context below.

Rules:
- Use ONLY the information in the provided context passages.
- Cite sources using bracket notation [1], [2], etc. corresponding to the passage numbers.
- If multiple passages support a statement, cite all of them, e.g. [1][3].
- If the context does not contain enough information, say so explicitly.
- Be concise but thorough.
- End your answer with a "Sources:" section listing the cited passage numbers and titles.

Available sources: {source_list}

---
Context:
{context}
---

Question: {question}

Answer:
"""