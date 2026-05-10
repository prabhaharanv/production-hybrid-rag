"""Parent-child chunking strategy.

Small chunks are used for retrieval precision (better embedding match),
but the full parent chunk is returned to the LLM for richer context.
This solves the chunk-size dilemma: small for search, large for generation.

Architecture:
- Documents are first split into "parent" chunks (e.g., 800 words)
- Each parent is further split into "child" chunks (e.g., 200 words)
- Child chunks are indexed and used for retrieval
- When a child matches, its parent chunk text is returned for generation
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.chunking import get_chunking_strategy


class ParentChildChunker:
    """Creates parent-child chunk relationships."""

    def __init__(
        self,
        parent_chunk_size: int = 800,
        parent_overlap: int = 100,
        child_chunk_size: int = 200,
        child_overlap: int = 30,
        parent_strategy: str = "word",
        child_strategy: str = "word",
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_strategy = get_chunking_strategy(parent_strategy)
        self.child_strategy = get_chunking_strategy(child_strategy)

    def chunk_documents(
        self, documents: list[dict]
    ) -> tuple[list[dict], dict[str, str]]:
        """Split documents into parent and child chunks.

        Returns:
            (child_chunks, child_to_parent_map)
            - child_chunks: list of child chunk records (for indexing)
            - child_to_parent_map: dict mapping child_chunk_id -> parent text
        """
        child_chunks = []
        child_to_parent: dict[str, str] = {}

        for doc in documents:
            # Create parent chunks
            parent_texts = self.parent_strategy.chunk(
                doc["text"],
                chunk_size=self.parent_chunk_size,
                overlap=self.parent_overlap,
            )

            for p_idx, parent_text in enumerate(parent_texts):
                parent_id = f"{doc['doc_id']}_parent_{p_idx}"

                # Create child chunks from this parent
                child_texts = self.child_strategy.chunk(
                    parent_text,
                    chunk_size=self.child_chunk_size,
                    overlap=self.child_overlap,
                )

                for c_idx, child_text in enumerate(child_texts):
                    child_id = f"{parent_id}_child_{c_idx}"
                    child_chunks.append(
                        {
                            "chunk_id": child_id,
                            "doc_id": doc["doc_id"],
                            "title": doc["title"],
                            "source": doc["source"],
                            "text": child_text,
                            "metadata": {
                                **doc.get("metadata", {}),
                                "parent_id": parent_id,
                                "child_index": c_idx,
                                "chunk_index": len(child_chunks),
                                "chunking_strategy": "parent_child",
                            },
                        }
                    )
                    child_to_parent[child_id] = parent_text

        return child_chunks, child_to_parent


class ParentChildStore:
    """Persists the child-to-parent mapping alongside the index."""

    def __init__(self, child_to_parent: dict[str, str] | None = None):
        self.child_to_parent = child_to_parent or {}

    def get_parent_text(self, child_chunk_id: str) -> str | None:
        return self.child_to_parent.get(child_chunk_id)

    def expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """Replace child chunk text with parent text for generation context."""
        seen_parents = set()
        expanded = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            parent_text = self.child_to_parent.get(chunk_id)

            if parent_text:
                parent_id = chunk.get("metadata", {}).get("parent_id", chunk_id)
                if parent_id in seen_parents:
                    continue  # Avoid duplicate parent text
                seen_parents.add(parent_id)
                expanded.append(
                    {
                        **chunk,
                        "text": parent_text,
                        "metadata": {
                            **chunk.get("metadata", {}),
                            "expanded_from_child": chunk_id,
                        },
                    }
                )
            else:
                expanded.append(chunk)

        return expanded

    def save(self, index_dir: str) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "parent_child_map.json", "w", encoding="utf-8") as f:
            json.dump(self.child_to_parent, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: str) -> "ParentChildStore":
        path = Path(index_dir) / "parent_child_map.json"
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return cls(mapping)
