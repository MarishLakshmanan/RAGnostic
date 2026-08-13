"""Chunking strategies that turn a PDF's raw bytes into a list of text chunks."""

from __future__ import annotations

from typing import Callable

import yaml
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

ChunkFn = Callable[[bytes], list[str]]

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["config"]


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract a PDF's full text, page by page, with embedded NUL bytes stripped.

    Args:
        pdf_bytes (bytes): The raw bytes of a PDF file to extract text from.

    Returns:
        str: The PDF's full text, with page breaks joined and NUL bytes removed.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        full_text = "\n\n".join([str(page.get_text("text")) for page in doc])
    finally:
        doc.close()

    # PDFs can embed literal NUL bytes in decoded text streams, which Postgres TEXT columns reject.
    return full_text.replace("\x00", "")


def chunk_pdf_bytes(
    pdf_bytes: bytes, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """Extract a PDF's text and split it into overlapping chunks for embedding/retrieval.

    pdf_bytes: the raw bytes of a PDF file to chunk.
    chunk_size: the target character length of each chunk.
    chunk_overlap: how many characters consecutive chunks share so context isn't lost at chunk boundaries.
    """
    full_text = extract_pdf_text(pdf_bytes)

    match config["chunking_strategy"]:
        case "recursive_chunker":
            from chunking.chunkers.recursive_chunker import recursive_chunk_text

            return recursive_chunk_text(full_text, chunk_size, chunk_overlap)
        case _:
            from chunking.chunkers.recursive_chunker import recursive_chunk_text

            return recursive_chunk_text(full_text, chunk_size, chunk_overlap)
