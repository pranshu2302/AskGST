"""
Test retrieval: embed queries, search Qdrant, print top 5 results per query.
"""
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "askgst_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
QUERIES = [
    "What is the GST registration threshold for small businesses?",
    "Do I need to register for GST if my turnover is 15 lakhs?",
    "What is reverse charge mechanism?",
    "Can I claim input tax credit on rent-a-cab services?",
    "When is GSTR-3B due for monthly filers?",
    "What is the GST rate on legal services?",
    "How does composition scheme work for traders?"
]


def main():
    print("Loading BGE model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    print("Connecting to Qdrant...")
    client = QdrantClient(host="localhost", port=6333)
    print("Connected.")

    for query in QUERIES:
        print("=" * 63)
        print(f"Query: {query}")
        print("=" * 63)
        prefixed_query = QUERY_PREFIX + query
        query_vector = model.encode(prefixed_query).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5
        ).points
        for hit in results:
            score = hit.score
            payload = hit.payload
            source = payload.get("source", "?")
            page = payload.get("page", "?")
            chunk_id = payload.get("id", "?")
            content = payload.get("content", "")
            preview = content[:250].replace("\n", " ").strip()
            print(f"  [Score: {score:.3f}] {source} (page {page}) — id: {chunk_id}")
            print(f"    Content preview: {preview}")
        print()


if __name__ == "__main__":
    main()
