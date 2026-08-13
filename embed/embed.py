"""Embed chunk rows from Postgres and upsert the vectors into Chroma."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import select, update

from clients.chroma_client import get_collection
from clients.embedding_client import embed_texts
from store.db import SessionLocal, init_db
from store.models import Chunk


def iter_unembedded_chunk_batches(batch_size: int = 32) -> Iterator[list[Chunk]]:
    """Yield batches of chunk rows that haven't been embedded yet.

    Args:
        batch_size (int, optional): How many chunks to fetch per batch. Defaults to 32,
            matching the embedding server's maximum batch size.

    Yields:
        list[Chunk]: A batch of chunk rows with embedded_at still NULL.
    """
    while True:
        with SessionLocal() as session:
            stmt = select(Chunk).where(Chunk.embedded_at.is_(None)).limit(batch_size)
            batch = session.execute(stmt).scalars().all()

        if not batch:
            return
        yield batch


def mark_chunks_embedded(chunk_ids: list[int]) -> None:
    """Record that a batch of chunks has been embedded.

    Args:
        chunk_ids (list[int]): The ids of the chunks to mark as embedded.
    """
    stmt = (
        update(Chunk)
        .where(Chunk.id.in_(chunk_ids))
        .values(embedded_at=datetime.now(timezone.utc))
    )
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()


def embed_and_upsert_batch(chunks: list[Chunk]) -> int:
    """Embed a batch of chunks, upsert the vectors into Chroma, then mark them embedded.

    Args:
        chunks (list[Chunk]): The chunk rows to embed.

    Returns:
        int: The number of chunks processed.
    """
    vectors = embed_texts([chunk.content for chunk in chunks])
    get_collection().upsert(
        ids=[str(chunk.id) for chunk in chunks],
        embeddings=vectors,
        metadatas=[
            {"paper_id": chunk.paper_id, "chunk_index": chunk.chunk_index} for chunk in chunks
        ],
    )
    mark_chunks_embedded([chunk.id for chunk in chunks])
    return len(chunks)


if __name__ == "__main__":
    init_db()

    total = 0
    for batch in iter_unembedded_chunk_batches():
        n = embed_and_upsert_batch(batch)
        total += n
        print(f"Embedded {n} chunks (running total: {total})")

    print(f"\nDone. {total} chunks embedded.")
