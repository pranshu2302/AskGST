"""
Centralized Qdrant client construction.

If QDRANT_URL is set in env, connects to that cluster (using QDRANT_API_KEY).
Otherwise falls back to local Docker Qdrant on localhost:6333.

This single source of truth lets the same code run locally (with Docker) or
in production (with Qdrant Cloud) by toggling env vars only.
"""
import os
from qdrant_client import QdrantClient


def get_qdrant_client():
    """Return a QdrantClient configured from env vars, with local fallback."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if url:
        # Production / cloud mode
        return QdrantClient(url=url, api_key=api_key)
    else:
        # Local fallback (Docker)
        return QdrantClient(host="localhost", port=6333)
