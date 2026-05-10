class DenseRetriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(query_vector, top_k=top_k)


class SparseRetriever:
    def __init__(self, bm25_store):
        self.bm25_store = bm25_store

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.bm25_store.search(query, top_k=top_k)


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse multiple ranked lists using RRF. Returns deduplicated results sorted by fused score."""
    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = {
                    key: val for key, val in item.items() if key != "score"
                }

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**chunk_map[cid], "score": score} for cid, score in ranked]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, rrf_k: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        dense_results = self.dense.retrieve(query, top_k=top_k)
        sparse_results = self.sparse.retrieve(query, top_k=top_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        return fused[:top_k]
