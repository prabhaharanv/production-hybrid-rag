"""HyDE (Hypothetical Document Embeddings) retrieval.

Generates a hypothetical answer to the query using the LLM, then embeds
that hypothetical document for retrieval. This dramatically improves recall
because the embedding space of a full-sentence answer is closer to stored
document chunks than a short query.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
"""

from openai import OpenAI


HYDE_PROMPT = (
    "Given the following question, write a short passage (2-4 sentences) that would "
    "directly answer it. Write as if you are writing a factual document. "
    "Do not say 'I don't know'. Always produce a plausible answer.\n\n"
    "Question: {question}\n\nPassage:"
)


class HyDEGenerator:
    """Generates a hypothetical document for embedding-based retrieval."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_hypothetical(self, question: str) -> str:
        """Generate a hypothetical answer passage for the given question."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable assistant that writes factual passages.",
                },
                {
                    "role": "user",
                    "content": HYDE_PROMPT.format(question=question),
                },
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()


class HyDERetriever:
    """Retriever that uses HyDE to improve dense retrieval recall.

    Generates a hypothetical document, embeds it, then uses that embedding
    to search the vector store — instead of embedding the raw query.
    """

    def __init__(self, hyde_generator: HyDEGenerator, embedder, vector_store):
        self.hyde_generator = hyde_generator
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        hypothetical_doc = self.hyde_generator.generate_hypothetical(query)
        hyde_vector = self.embedder.embed_query(hypothetical_doc)
        results = self.vector_store.search(hyde_vector, top_k=top_k)
        # Attach the hypothetical doc for transparency
        for r in results:
            r["hyde_doc"] = hypothetical_doc
        return results
