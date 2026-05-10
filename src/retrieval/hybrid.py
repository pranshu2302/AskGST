"""
Hybrid retrieval combining BGE vector search and BM25 keyword search using
Reciprocal Rank Fusion (RRF), with optional cross-encoder reranking.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.retrieval.bm25_store import BM25Store
from src.retrieval.qdrant_factory import get_qdrant_client

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "askgst_chunks"
RRF_K = 60  # smoothing constant


class HybridRetriever:
    def __init__(self, bm25_store, vector_model, qdrant_client, reranker=None):
        self.bm25_store = bm25_store
        self.vector_model = vector_model
        self.qdrant_client = qdrant_client
        self.reranker = reranker  # Optional Reranker instance

    def _vector_search(self, query, top_n):
        """Run BGE + Qdrant search."""
        prefixed = QUERY_PREFIX + query
        vector = self.vector_model.encode(prefixed).tolist()
        results = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_n
        ).points
        return [(hit.payload, hit.score) for hit in results]

    def _bm25_search(self, query, top_n):
        """Run BM25 search."""
        return self.bm25_store.search(query, top_k=top_n)

    def _rrf_merge(self, vector_results, bm25_results, top_k):
        """Reciprocal Rank Fusion merge."""
        merged = {}
        # Vector results
        for rank, (chunk, _score) in enumerate(vector_results, start=1):
            chunk_id = chunk["id"]
            if chunk_id not in merged:
                merged[chunk_id] = {"chunk": chunk, "rrf_score": 0.0}
            merged[chunk_id]["rrf_score"] += 1.0 / (RRF_K + rank)
        # BM25 results
        for rank, (chunk, _score) in enumerate(bm25_results, start=1):
            chunk_id = chunk["id"]
            if chunk_id not in merged:
                merged[chunk_id] = {"chunk": chunk, "rrf_score": 0.0}
            merged[chunk_id]["rrf_score"] += 1.0 / (RRF_K + rank)
        # Sort and return
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]
        return [(item["chunk"], item["rrf_score"]) for item in sorted_results]

    def search(self, query, top_k=5, candidates_per_retriever=20, rerank_pool=20):
        """
        Hybrid retrieval with optional reranking.

        Args:
            query: user question
            top_k: final number of results to return
            candidates_per_retriever: top N to take from each of BM25 and vector
            rerank_pool: candidates to pass to the reranker (only used if reranker is set).
                         Should be >= top_k so reranker has options to choose from.

        Returns:
            list of (chunk_dict, score) tuples
        """
        vector_results = self._vector_search(query, candidates_per_retriever)
        bm25_results = self._bm25_search(query, candidates_per_retriever)

        if self.reranker is not None:
            # Get a larger merged pool, then rerank to top_k
            merged_pool = self._rrf_merge(vector_results, bm25_results, top_k=rerank_pool)
            return self.reranker.rerank(query, merged_pool, top_k=top_k)
        else:
            # No reranker: just RRF merge to final top_k
            return self._rrf_merge(vector_results, bm25_results, top_k=top_k)


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from dotenv import load_dotenv
    from src.retrieval.reranker import Reranker

    load_dotenv()

    print("Initializing components...")
    bm25 = BM25Store("data/processed/chunks.json")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = get_qdrant_client()
    reranker = Reranker()

    # Two retrievers — one with reranker, one without
    hybrid_no_rerank = HybridRetriever(bm25, model, client)
    hybrid_with_rerank = HybridRetriever(bm25, model, client, reranker=reranker)

    test_queries = [
        "What is the GST registration threshold for small businesses?",
        "What is the GST rate on legal services?",
        "Can I claim input tax credit on rent-a-cab services?",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        print("\n--- Without reranker (pure RRF) ---")
        results = hybrid_no_rerank.search(query, top_k=5)
        for i, (chunk, score) in enumerate(results, 1):
            page = chunk.get("page", "—")
            print(f"  [{i}] {score:.4f} | {chunk['source']} (p.{page}) — {chunk['id']}")

        print("\n--- With reranker ---")
        results = hybrid_with_rerank.search(query, top_k=5)
        for i, (chunk, score) in enumerate(results, 1):
            page = chunk.get("page", "—")
            print(f"  [{i}] {score:+.3f} | {chunk['source']} (p.{page}) — {chunk['id']}")

        print()