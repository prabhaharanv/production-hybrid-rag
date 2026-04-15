import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, records: list[dict] | None = None):
        self.records: list[dict] = records or []
        self.bm25: BM25Okapi | None = None

        if self.records:
            self._build_index()

    def _build_index(self) -> None:
        corpus = [_tokenize(r["text"]) for r in self.records]
        self.bm25 = BM25Okapi(corpus)

    def add(self, records: list[dict]) -> None:
        self.records.extend(records)
        self._build_index()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None or not self.records:
            return []

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                break
            print(f"results before: {results}")
            results.append({**self.records[idx], "score": float(score)})
            print(f"results after: {results}")

        return results

    def save(self, index_dir: str) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        with open(path / "bm25_records.json", "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "BM25Store":
        path = Path(index_dir)

        with open(path / "bm25_records.json", "r", encoding="utf-8") as f:
            records = json.load(f)

        with open(path / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)  # noqa: S301 — trusted local index file

        store = cls.__new__(cls)
        store.records = records
        store.bm25 = bm25
        return store
