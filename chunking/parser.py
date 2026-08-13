"""Walk the `papers` table, chunk each reachable PDF, and persist the chunks."""

from __future__ import annotations

from typing import Iterator

import requests
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from chunking.bibliography_filter import filter_bibliography_chunks
from chunking.chunker import ChunkFn, chunk_pdf_bytes
from corpus.fetch_corpus import _is_pdf_reachable
from store.db import SessionLocal
from store.models import Chunk, PaperRecord


def iter_paper_chunks(
    chunk_fn: ChunkFn = chunk_pdf_bytes, timeout: float = 10.0
) -> Iterator[tuple[PaperRecord, list[str]]]:
    """Yield each paper in the DB paired with its chunks, skipping papers whose PDF isn't reachable.

    chunk_fn: which chunking strategy to use — defaults to the PyMuPDF + RecursiveCharacterTextSplitter implementation.
    timeout: how many seconds to wait for the PDF reachability check and download before giving up.
    """
    session_ = requests.Session()

    with SessionLocal() as session:
        papers = session.execute(select(PaperRecord)).scalars()
        for paper in papers:
            if not _is_pdf_reachable(paper.pdf_url, session_, timeout):
                continue
            pdf_bytes = session_.get(paper.pdf_url, timeout=timeout).content
            chunks = chunk_fn(pdf_bytes)
            chunks = filter_bibliography_chunks(pdf_bytes, chunks)
            yield paper, chunks


def save_chunks_to_db(paper_id: int, chunks: list[str], strategy: str) -> int:
    """Bulk-insert a paper's chunks, skipping any that already exist for that paper and strategy.

    paper_id: the id of the paper these chunks belong to.
    chunks: the ordered list of chunk texts to insert.
    strategy: name of the chunking strategy that produced these chunks.
    """
    if not chunks:
        return 0

    rows = [
        {"paper_id": paper_id, "chunk_index": i, "content": text, "strategy": strategy}
        for i, text in enumerate(chunks)
    ]
    stmt = (
        insert(Chunk)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["paper_id", "chunk_index", "strategy"])
        .returning(Chunk.id)
    )

    with SessionLocal() as session:
        result = session.execute(stmt)
        session.commit()
        return len(result.fetchall())


if __name__ == "__main__":
    from store.db import init_db

    init_db()

    # This where I can switch chunking strategies if I want to try different ones.
    chunk_fn = chunk_pdf_bytes
    strategy = chunk_fn.__name__

    total_chunks = 0
    for paper, chunks in iter_paper_chunks(chunk_fn):
        inserted = save_chunks_to_db(paper.id, chunks, strategy)
        total_chunks += inserted
        print(f"[{paper.arxiv_id}] {len(chunks)} chunks extracted, {inserted} inserted")

    print(f"\nDone. {total_chunks} chunks inserted.")
