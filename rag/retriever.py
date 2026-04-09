class Retriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(query_vector, top_k=top_k)