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


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]