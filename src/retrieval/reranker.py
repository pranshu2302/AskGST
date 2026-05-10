"""
Cross-encoder reranking using BAAI/bge-reranker-base.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self):
        """Load the cross-encoder model."""
        print("Loading reranker model...")
        self.model = CrossEncoder(RERANKER_MODEL_NAME)
        print("Reranker ready.")
    
    def rerank(self, query, candidates, top_k=5):
        """
        Re-score and sort candidates by query relevance.
        
        Args:
            query: the user's question
            candidates: list of (chunk_dict, original_score) tuples (e.g., from hybrid retrieval)
            top_k: how many top candidates to return
        
        Returns:
            list of (chunk_dict, rerank_score) tuples sorted by rerank_score descending
        """
        if not candidates:
            return []
        
        # Build (query, content) pairs for the cross-encoder
        pairs = [(query, chunk["content"]) for chunk, _orig_score in candidates]
        
        # Score them all in one batch
        scores = self.model.predict(pairs)
        
        # Pair each candidate with its new score
        rescored = [
            (candidates[i][0], float(scores[i]))
            for i in range(len(candidates))
        ]
        
        # Sort by rerank score descending, take top K
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]

if __name__ == "__main__":
    print("Initializing components...")
    bm25 = BM25Store("data/processed/chunks.json")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client = QdrantClient(host="localhost", port=6333)
    hybrid = HybridRetriever(bm25, model, client)
    reranker = Reranker()
    
    test_queries = [
        "What is the GST registration threshold for small businesses?",
        "What is the GST rate on legal services?",          # the weak query from Day 3
        "Can I claim input tax credit on rent-a-cab services?",
        "How does composition scheme work for traders?",
    ]
    
    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        
        # Get top 20 from hybrid (the candidate pool)
        candidates = hybrid.search(query, top_k=20, candidates_per_retriever=20)
        
        # Rerank to top 5
        reranked = reranker.rerank(query, candidates, top_k=5)
        
        print("\nReranked top 5:")
        for i, (chunk, score) in enumerate(reranked, start=1):
            page = chunk.get("page", "—")
            print(f"  [{i}] Rerank: {score:+.3f} | {chunk['source']} (page {page}) — id: {chunk['id']}")
            preview = chunk["content"][:200].replace("\n", " ").strip()
            print(f"      {preview}")
        print()
