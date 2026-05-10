"""
Eval harness — runs 4 retrieval configurations against the test queries
in src/eval/queries.json and computes recall@5 per config.

Configurations:
  vector_only    : BGE + Qdrant, top 5
  bm25_only      : BM25Store.search, top 5
  hybrid         : HybridRetriever (no reranker), top 5
  hybrid_rerank  : HybridRetriever (with reranker), top 5

Output: data/eval/results.json
"""
import sys
from pathlib import Path

# Make `src.*` imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import time

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

# --- Constants ---
REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = REPO_ROOT / "src" / "eval" / "queries.json"
RESULTS_PATH = REPO_ROOT / "data" / "eval" / "results.json"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.json"

COLLECTION_NAME = "askgst_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 5


# --- Retrieval functions (one per config) ---

def retrieve_vector_only(query, model, qdrant_client):
    """Pure dense retrieval: BGE-encoded query against Qdrant. Returns chunk IDs."""
    prefixed = QUERY_PREFIX + query
    vector = model.encode(prefixed).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=TOP_K,
    ).points
    return [hit.payload["id"] for hit in results]


def retrieve_bm25_only(query, bm25_store):
    """Pure sparse retrieval: BM25 over tokenized chunks. Returns chunk IDs."""
    results = bm25_store.search(query, top_k=TOP_K)
    return [chunk["id"] for chunk, _score in results]


def retrieve_hybrid(query, hybrid_retriever):
    """RRF merge of dense+sparse, optionally with reranking (depends on retriever)."""
    results = hybrid_retriever.search(query, top_k=TOP_K)
    return [chunk["id"] for chunk, _score in results]


# --- Evaluation logic ---

def evaluate_query(query_entry, retrievers):
    """Run all configs for one query, return per-config retrieval + pass/fail."""
    query = query_entry["query"]
    gold_set = set(query_entry["gold_chunk_ids"])

    retrieved = {}
    for config_name, retrieve_fn in retrievers.items():
        chunk_ids = retrieve_fn(query)
        passed = bool(gold_set & set(chunk_ids))  # any gold chunk in top 5?
        retrieved[config_name] = {
            "chunk_ids": chunk_ids,
            "passed": passed,
        }

    return {
        "query_id": query_entry["id"],
        "query": query_entry["query"],
        "gold_chunk_ids": query_entry["gold_chunk_ids"],
        "category": query_entry["category"],
        "phrasing_style": query_entry["phrasing_style"],
        "retrieved": retrieved,
    }


def compute_summary(per_query_results, configs):
    """Recall@5 per config, computed only over queries that have gold chunks."""
    queries_with_gold = [r for r in per_query_results if r["gold_chunk_ids"]]
    n = len(queries_with_gold)

    metrics_per_config = {}
    for config in configs:
        passed = sum(1 for r in queries_with_gold if r["retrieved"][config]["passed"])
        metrics_per_config[config] = {
            "recall_at_5": round(passed / n, 4) if n > 0 else 0.0,
            "passed": passed,
            "total_with_gold": n,
        }

    return {
        "total_queries": len(per_query_results),
        "queries_with_gold": n,
        "metrics_per_config": metrics_per_config,
    }


# --- Main ---

def main():
    # Load queries
    with open(QUERIES_PATH) as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")

    # Initialize components ONCE (shared across all 30 queries x 4 configs)
    print("\nInitializing components...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    qdrant_client = QdrantClient(host="localhost", port=6333)
    bm25_store = BM25Store(str(CHUNKS_PATH))
    reranker = Reranker()
    hybrid_no_rerank = HybridRetriever(bm25_store, model, qdrant_client)
    hybrid_with_rerank = HybridRetriever(bm25_store, model, qdrant_client, reranker=reranker)
    print(f"Initialization done in {time.time() - t0:.1f}s")

    # Map config name -> single-arg retrieval callable
    retrievers = {
        "vector_only":   lambda q: retrieve_vector_only(q, model, qdrant_client),
        "bm25_only":     lambda q: retrieve_bm25_only(q, bm25_store),
        "hybrid":        lambda q: retrieve_hybrid(q, hybrid_no_rerank),
        "hybrid_rerank": lambda q: retrieve_hybrid(q, hybrid_with_rerank),
    }

    # Run eval
    print(f"\nRunning {len(retrievers)} configs on {len(queries)} queries...\n")
    eval_start = time.time()
    per_query_results = []
    for i, q_entry in enumerate(queries, start=1):
        preview = q_entry["query"][:60] + ("..." if len(q_entry["query"]) > 60 else "")
        print(f"[{i:2d}/{len(queries)}] {q_entry['id']}: {preview}")
        result = evaluate_query(q_entry, retrievers)
        per_query_results.append(result)
    eval_elapsed = time.time() - eval_start

    # Summary
    summary = compute_summary(per_query_results, list(retrievers.keys()))

    # Save BEFORE printing — defensive (don't lose results to a print error)
    output = {"summary": summary, "per_query_results": per_query_results}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Total queries:     {summary['total_queries']}")
    print(f"Queries with gold: {summary['queries_with_gold']}")
    print(f"Eval runtime:      {eval_elapsed:.1f}s")
    print("\nRecall@5 by configuration:")
    for config, metrics in summary["metrics_per_config"].items():
        print(
            f"  {config:15s}: {metrics['recall_at_5']:.2%} "
            f"({metrics['passed']}/{metrics['total_with_gold']})"
        )
    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
