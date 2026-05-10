"""
One-time migration: copy all points from local Qdrant (Docker) to Qdrant Cloud.

Run this after creating a Qdrant Cloud cluster and adding QDRANT_URL +
QDRANT_API_KEY to your .env file. Both the local Docker Qdrant and the cloud
cluster need to be reachable.

Usage:
    python scripts/migrate_to_cloud.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load .env from project root regardless of where the script is run from
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LOCAL_HOST = "localhost"
LOCAL_PORT = 6333
CLOUD_URL = os.getenv("QDRANT_URL")
CLOUD_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "askgst_chunks"
BATCH_SIZE = 64
VECTOR_SIZE = 384


def main() -> int:
    if not CLOUD_URL or not CLOUD_API_KEY:
        print("ERROR: QDRANT_URL or QDRANT_API_KEY not set in .env", file=sys.stderr)
        print("       Add both to your .env file before running this script.", file=sys.stderr)
        return 1

    # 1. Connect to local Qdrant
    print("Connecting to local Qdrant (Docker)...")
    local = QdrantClient(host=LOCAL_HOST, port=LOCAL_PORT)
    try:
        local_info = local.get_collection(COLLECTION_NAME)
    except (UnexpectedResponse, ValueError) as e:
        print(f"ERROR: local collection '{COLLECTION_NAME}' not found.", file=sys.stderr)
        print(f"       Run `python src/retrieval/vector_store.py` first.", file=sys.stderr)
        print(f"       Underlying error: {e}", file=sys.stderr)
        return 1
    total = local_info.points_count
    print(f"Local collection has {total} points.")

    # 2. Connect to cloud Qdrant
    print("Connecting to Qdrant Cloud...")
    cloud = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)
    # Force a real network call up front so auth/URL errors fail loudly here
    # rather than later inside the upsert loop.
    cloud.get_collections()

    # 3. Create the collection on cloud (delete first if it exists; ignore "not found")
    print(f"Creating collection '{COLLECTION_NAME}' on cloud...")
    try:
        cloud.delete_collection(COLLECTION_NAME)
        print("  (deleted existing cloud collection)")
    except UnexpectedResponse as e:
        # 404 = collection didn't exist yet; anything else is a real problem
        if e.status_code != 404:
            raise
    cloud.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # 4. Scroll through local in batches, upload to cloud
    print("Migrating points...")
    offset = None
    migrated = 0
    while True:
        points, next_offset = local.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        # Sanity-check: this script assumes single unnamed vectors (matches
        # vector_store.py's VectorParams config). If a dict shows up here, the
        # local collection was created with named vectors and this script needs
        # tweaking — fail loudly rather than silently corrupting the upload.
        first_vec = points[0].vector
        if isinstance(first_vec, dict):
            print(
                "ERROR: local collection uses named vectors (dict shape). "
                "This migration script assumes single unnamed vectors. "
                "Adjust VectorParams + PointStruct to match.",
                file=sys.stderr,
            )
            return 1

        cloud_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        cloud.upsert(collection_name=COLLECTION_NAME, points=cloud_points)
        migrated += len(cloud_points)
        print(f"  Migrated {migrated}/{total}")

        if next_offset is None:
            break
        offset = next_offset

    # 5. Verify
    cloud_info = cloud.get_collection(COLLECTION_NAME)
    print(f"\nMigration complete. Cloud collection has {cloud_info.points_count} points.")
    if cloud_info.points_count != total:
        print(
            f"⚠️  Point count mismatch: local={total}, cloud={cloud_info.points_count}",
            file=sys.stderr,
        )
        return 1
    print("✅ All points verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())