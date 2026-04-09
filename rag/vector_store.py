import faiss
import json
import numpy as np
from pathlib import Path


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.records: list[dict] = []

    def add(self, vectors: np.ndarray, records: list[dict]) -> None:
        if len(vectors) != len(records):
            raise ValueError("vectors and records length mismatch")

        self.index.add(vectors)
        self.records.extend(records)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        scores, indices = self.index.search(query_vector, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            record = self.records[idx]
            results.append(
                {
                    **record,
                    "score": float(score),
                }
            )

        return results

    def save(self, index_dir: str) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        with open(path / "records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str):
        path = Path(index_dir)
        index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "records.json", "r", encoding="utf-8") as f:
            records = json.load(f)

        store = cls(dim=index.d)
        store.index = index
        store.records = records
        return store