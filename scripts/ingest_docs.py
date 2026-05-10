import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from rag.ingest import run_ingestion


if __name__ == "__main__":
    result = run_ingestion(
        raw_data_dir=settings.raw_data_dir,
        index_dir=settings.index_dir,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunking_strategy=settings.chunking_strategy,
    )
    print(result)
