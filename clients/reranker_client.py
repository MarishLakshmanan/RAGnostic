"""HTTP client for the self-hosted cross-encoder reranker."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

load_dotenv()

CROSS_ENCODER_URL = os.environ["CROSS_ENCODER_URL"]


def rerank(query: str, texts: list[str]) -> list[dict]:
    """Score and rank a list of texts against a query using the cross-encoder server.

    Args:
        query (str): The query to score texts against.
        texts (list[str]): The candidate texts to rerank.

    Returns:
        list[dict]: Each item has "index" (int, position in `texts`) and "score" (float),
            sorted by score descending.
    """
    response = requests.post(
        f"{CROSS_ENCODER_URL}/rerank", json={"query": query, "texts": texts}
    )
    response.raise_for_status()
    return response.json()
