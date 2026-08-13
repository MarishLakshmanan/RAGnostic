"""HTTP client for the OpenAI-compatible embedding server."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

client = OpenAI(base_url=os.environ["EMBEDDING_BASE_URL"], api_key="not-needed")
async_client = AsyncOpenAI(
    base_url=os.environ["EMBEDDING_BASE_URL"], api_key="not-needed"
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured embedding model.

    Args:
        texts (list[str]): The chunk texts to embed.

    Returns:
        list[list[float]]: The embedding vectors, in the same order as the input texts.
    """
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in response.data]
