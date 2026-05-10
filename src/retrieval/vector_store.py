c"""
Loads chunks.json, generates 384-dim embeddings for each chunk, and uploads to Qdrant.
"""
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

BATCH_SIZE = 64
CHUNKS_PATH = Path("data/processed/chunks.json")
COLLECTION_NAME = "askgst_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    print("Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    print("Initializing BGE model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model ready.")

    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print(f"Recreating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Note: BGE recommends a query prefix at search time, not for indexing.
    # We embed chunks as-is here. Query prefix is applied in retrieval module.
    total = len(chunks)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        vectors = model.encode(texts, show_progress_bar=False)
        points = []
        for i, chunk in enumerate(batch):
            points.append(PointStruct(
                id=batch_start + i,
                vector=vectors[i].tolist(),
                payload=chunk
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Embedded and upserted: {min(batch_start + len(batch), total)}/{total}")

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {info.points_count} points.")


if __name__ == "__main__":
    main()
