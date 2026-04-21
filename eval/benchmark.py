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
from eval.metrics import evaluate_single


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


def run_benchmark(eval_path: str, top_k: int = 5, enable_deep_eval: bool = False) -> dict:
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

    # Accumulators for deep eval metrics (only when enabled)
    deep_accum = {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "bertscore_f1": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "hallucination_rate": 0.0,
    }
    deep_count = 0

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

        # Deep evaluation metrics
        if enable_deep_eval and not response["abstained"]:
            context_text = "\n".join(
                c.get("text", "") for c in response.get("retrieved_chunks", [])
            )
            deep = evaluate_single(
                question=item["question"],
                answer=response["answer"],
                context=context_text,
                retrieved_chunks=response.get("retrieved_chunks", []),
                ground_truth=item.get("ground_truth"),
                relevant_source=item.get("expected_source"),
                k=top_k,
            )
            result["deep_eval"] = deep
            for key in deep_accum:
                val = deep.get(key)
                if val is not None:
                    deep_accum[key] += val
            deep_count += 1

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

    if enable_deep_eval and deep_count > 0:
        summary["deep_eval"] = {
            key: round(val / deep_count, 4) for key, val in deep_accum.items()
        }

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Benchmark Suite")
    parser.add_argument("eval_path", nargs="?", default="eval/dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--deep-eval", action="store_true", help="Enable deep evaluation metrics (RAGAS, BERTScore, MRR, NDCG, hallucination)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print(f"Running benchmark on: {args.eval_path}")
    if args.deep_eval:
        print("Deep evaluation metrics ENABLED (this will be slower)\n")
    else:
        print("Deep evaluation metrics disabled (use --deep-eval to enable)\n")

    report = run_benchmark(args.eval_path, top_k=args.top_k, enable_deep_eval=args.deep_eval)

    print("\n=== Benchmark Summary ===")
    for key, val in report["summary"].items():
        if key == "deep_eval":
            print("  --- Deep Eval Averages ---")
            for dk, dv in val.items():
                print(f"    {dk}: {dv}")
        else:
            print(f"  {key}: {val}")

    output_path = "eval/results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {output_path}")
