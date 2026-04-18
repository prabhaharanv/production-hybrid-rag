from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "data/raw")
    processed_dir: str = os.getenv("PROCESSED_DIR", "data/processed")
    index_dir: str = os.getenv("INDEX_DIR", "data/index")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    enable_reranker: bool = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
    enable_query_rewriting: bool = os.getenv("ENABLE_QUERY_REWRITING", "true").lower() == "true"
    rag_api_key: str | None = os.getenv("RAG_API_KEY")
    rate_limit: str = os.getenv("RATE_LIMIT", "20/minute")


settings = Settings()