"""HTTP client for the self-hosted ChromaDB vector store."""

from __future__ import annotations

import os

from chromadb import HttpClient
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

load_dotenv()

CHROMA_COLLECTION = os.environ["CHROMA_COLLECTION"]

client = HttpClient(host=os.environ["CHROMA_HOST"], port=int(os.environ["CHROMA_PORT"]))


def get_collection() -> Collection:
    """Return the Chroma collection that stores chunk embeddings, creating it if it doesn't exist.

    Returns:
        Collection: the Chroma collection configured for this project's chunk vectors.
    """
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={
            "embedding_model": os.environ["EMBEDDING_MODEL"],
            "topic": "Cybersecurity in LLM ops",
        },
    )
