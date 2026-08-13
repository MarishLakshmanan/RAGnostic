"""Build a BM25 sparse index over chunk text and persist it to disk."""

from __future__ import annotations

import bm25s
from sqlalchemy import select

from store.db import SessionLocal, init_db
from store.models import Chunk

BM25_INDEX_DIR = "store/bm25_index"


def build_bm25_index(
    strategy: str = "chunk_pdf_bytes", index_dir: str = BM25_INDEX_DIR
) -> int:
    """Build a BM25 index over a chunking strategy's chunks and save it to disk.

    Args:
        strategy (str, optional): Which chunking strategy's chunk rows to index. Defaults to "chunk_pdf_bytes".
        index_dir (str, optional): Directory to save the index to. Defaults to "store/bm25_index".

    Returns:
        int: The number of chunks indexed.

    Raises:
        ValueError: If no chunks exist for the given strategy.
    """
    with SessionLocal() as session:
        chunks = (
            session.execute(select(Chunk).where(Chunk.strategy == strategy))
            .scalars()
            .all()
        )

    if not chunks:
        raise ValueError(
            f"No chunks found for strategy {strategy!r} - run chunking.parser first."
        )

    corpus = [{"id": chunk.id, "text": chunk.content} for chunk in chunks]
    tokens = bm25s.tokenize([row["text"] for row in corpus], stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save(index_dir, corpus=corpus)

    return len(corpus)


if __name__ == "__main__":
    init_db()
    count = build_bm25_index()
    print(f"Indexed {count} chunks into {BM25_INDEX_DIR}")
