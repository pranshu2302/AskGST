"""
BM25 keyword retrieval over GST chunks.
"""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, chunks_path):
        """
        Load chunks from JSON and build the BM25 index.
        """
        # 1. Load chunks
        with open(chunks_path) as f:
            self.chunks = json.load(f)
        
        # 2. Tokenize each chunk's content
        tokenized_corpus = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk["content"])
            tokenized_corpus.append(tokens)
        
        # 3. Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built: {len(self.chunks)} chunks indexed")

    def _tokenize(self, text):
        """Lowercase, strip punctuation (except hyphens)."""
        tokens = text.lower().split()
        tokens = [re.sub(r'[^\w\-]', '', t) for t in tokens]  # keep word chars + hyphens
        return [t for t in tokens if t]

    def search(self, query, top_k=20):
        """
        Returns a list of (chunk_dict, score) tuples sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

if __name__ == "__main__":
    store = BM25Store("data/processed/chunks.json")
    
    test_queries = [
        "GSTR-3B due date for monthly filers",
        "rent-a-cab input tax credit",
        "reverse charge mechanism",
        "GST registration threshold 20 lakhs",
        "Section 22 registration",
    ]
    
    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
        results = store.search(query, top_k=5)
        for i, (chunk, score) in enumerate(results, 1):
            page = chunk.get("page", "—")
            print(f"  [{i}] Score: {score:.3f} | {chunk['source']} (page {page}) — id: {chunk['id']}")
            preview = chunk["content"][:200].replace("\n", " ").strip()
            print(f"      {preview}")
        print()
