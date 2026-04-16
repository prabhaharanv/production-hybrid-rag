import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from rag.embeddings import SentenceTransformerEmbedder
from rag.vector_store import FaissVectorStore
from rag.bm25_retriever import BM25Store
from rag.retriever import DenseRetriever, SparseRetriever, HybridRetriever
from rag.reranker import Reranker
from rag.query_rewriter import QueryRewriter
from rag.generator import LLMGenerator
from rag.pipeline import RAGPipeline


def load_eval_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def source_hit(citations: list[dict], expected_source: str | None) -> bool:
    if expected_source is None:
        return True
    return any(expected_source in c.get("title", "") or expected_source in c.get("source", "") for c in citations)


def run_benchmark(eval_path: str, top_k: int = 5) -> dict:
    dataset = load_eval_dataset(eval_path)

    # Build pipeline
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    vector_store = FaissVectorStore.load(settings.index_dir)
    bm25_store = BM25Store.load(settings.index_dir)

    dense = DenseRetriever(embedder=embedder, vector_store=vector_store)
    sparse = SparseRetriever(bm25_store=bm25_store)
    retriever = HybridRetriever(dense=dense, sparse=sparse, rrf_k=settings.rrf_k)

    reranker = Reranker(settings.reranker_model) if settings.enable_reranker else None
    query_rewriter = (
        QueryRewriter(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        if settings.enable_query_rewriting
        else None
    )
    generator = LLMGenerator(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        reranker=reranker,
        query_rewriter=query_rewriter,
    )

    results = []
    total_keyword_recall = 0.0
    total_source_hits = 0
    total_abstention_correct = 0
    total_latency = 0.0

    for item in dataset:
        start = time.time()
        response = pipeline.ask(item["question"], top_k=top_k)
        latency = time.time() - start

        kr = keyword_recall(response["answer"], item["expected_keywords"])
        sh = source_hit(response["citations"], item.get("expected_source"))
        abstention_correct = response["abstained"] == item["should_abstain"]

        total_keyword_recall += kr
        total_source_hits += int(sh)
        total_abstention_correct += int(abstention_correct)
        total_latency += latency

        result = {
            "id": item["id"],
            "question": item["question"],
            "rewritten_query": response["rewritten_query"],
            "answer": response["answer"][:200],
            "abstained": response["abstained"],
            "expected_abstain": item["should_abstain"],
            "abstention_correct": abstention_correct,
            "keyword_recall": round(kr, 2),
            "source_hit": sh,
            "latency_s": round(latency, 2),
        }
        results.append(result)
        print(f"  {item['id']}: keyword_recall={kr:.2f} source_hit={sh} abstention_ok={abstention_correct} latency={latency:.2f}s")

    n = len(dataset)
    summary = {
        "total_questions": n,
        "avg_keyword_recall": round(total_keyword_recall / n, 3),
        "source_hit_rate": round(total_source_hits / n, 3),
        "abstention_accuracy": round(total_abstention_correct / n, 3),
        "avg_latency_s": round(total_latency / n, 2),
        "total_latency_s": round(total_latency, 2),
    }

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    eval_path = sys.argv[1] if len(sys.argv) > 1 else "eval/dataset.json"
    print(f"Running benchmark on: {eval_path}\n")

    report = run_benchmark(eval_path)

    print("\n=== Benchmark Summary ===")
    for key, val in report["summary"].items():
        print(f"  {key}: {val}")

    output_path = "eval/results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {output_path}")
