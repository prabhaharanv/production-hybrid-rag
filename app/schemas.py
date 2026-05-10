from pydantic import BaseModel
from typing import List, Dict, Any


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class Citation(BaseModel):
    reference: int
    title: str
    source: str


class AskResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    abstained: bool
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
