"""Loader for the on-disk BM25 sparse index."""

from __future__ import annotations

import bm25s

BM25_INDEX_DIR = "store/bm25_index"

_retriever: bm25s.BM25 | None = None


def get_bm25_retriever() -> bm25s.BM25:
    """Load the BM25 index from disk, caching it after the first call.

    Returns:
        bm25s.BM25: The loaded retriever, with its corpus available for lookups.
    """
    global _retriever
    if _retriever is None:
        _retriever = bm25s.BM25.load(BM25_INDEX_DIR, load_corpus=True)
    return _retriever
