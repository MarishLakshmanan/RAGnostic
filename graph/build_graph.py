"""Build a knowledge graph of related chunks by querying Chroma for near neighbors."""

from __future__ import annotations

from typing import Iterator

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from clients.chroma_client import get_collection
from clients.embedding_client import embed_texts
from store.db import SessionLocal, init_db
from store.models import Chunk, Edge


def iter_embedded_chunk_batches(batch_size: int = 32) -> Iterator[list[Chunk]]:
    """Yield batches of embedded chunks that don't have an edge row yet.

    Args:
        batch_size (int, optional): How many chunks to fetch per batch. Defaults to 32.

    Yields:
        list[Chunk]: A batch of embedded chunk rows with no corresponding row in `edges`.
    """
    while True:
        with SessionLocal() as session:
            stmt = (
                select(Chunk)
                .where(Chunk.embedded_at.is_not(None))
                .where(~Chunk.id.in_(select(Edge.source_chunk_id)))
                .limit(batch_size)
            )
            batch = session.execute(stmt).scalars().all()

        if not batch:
            return
        yield batch


def build_and_save_edges_batch(
    chunks: list[Chunk], top_n: int = 5, min_index_gap: int = 3
) -> int:
    """Find near-neighbor chunks for a batch of chunks and persist them as edges.

    Args:
        chunks (list[Chunk]): The chunk rows to find neighbors for.
        top_n (int, optional): How many neighbors to keep per chunk. Defaults to 5.
        min_index_gap (int, optional): Minimum chunk_index distance a same-paper chunk
            must have to count as a neighbor, so trivially-similar adjacent chunks are
            excluded. Defaults to 3.

    Returns:
        int: The number of edge rows written.
    """
    vectors = embed_texts([chunk.content for chunk in chunks])

    rows = []
    for chunk, vector in zip(chunks, vectors):
        where = {
            "$or": [
                {"paper_id": {"$ne": chunk.paper_id}},
                {"chunk_index": {"$lt": chunk.chunk_index - min_index_gap}},
                {"chunk_index": {"$gt": chunk.chunk_index + min_index_gap}},
            ]
        }
        result = get_collection().query(query_embeddings=[vector], n_results=top_n, where=where)
        target_ids = [int(chunk_id) for chunk_id in result["ids"][0]]
        rows.append({"source_chunk_id": chunk.id, "target_chunk_ids": target_ids})

    if not rows:
        return 0

    stmt = (
        insert(Edge)
        .values(rows)
        .on_conflict_do_update(
            index_elements=["source_chunk_id"],
            set_={"target_chunk_ids": insert(Edge).excluded.target_chunk_ids},
        )
    )
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()
    return len(rows)


if __name__ == "__main__":
    init_db()

    total = 0
    for batch in iter_embedded_chunk_batches():
        n = build_and_save_edges_batch(batch)
        total += n
        print(f"Built edges for {n} chunks (running total: {total})")

    print(f"\nDone. Edges built for {total} chunks.")
