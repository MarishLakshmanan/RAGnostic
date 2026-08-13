"""Drop reference-list/bibliography chunks before they're persisted.

Locates a paper's own reference section (best-effort, via PDF outline or a heading
scan), embeds it in small fragments, and filters out any candidate chunk whose
embedding is a close match to one of those fragments. Papers where no reference
section can be located pass through unfiltered.
"""

from __future__ import annotations

import re

import numpy as np
import pymupdf
import yaml

from chunking.chunkers.recursive_chunker import recursive_chunk_text
from clients.embedding_client import embed_texts

with open("config.yaml", "r") as f:
    _bibliography_filter_config = yaml.safe_load(f)["config"]["bibliography_filter"]

SIMILARITY_THRESHOLD = _bibliography_filter_config["similarity_threshold"]

MAX_EMBED_BATCH_SIZE = 32  # matches the embedding server's maximum batch size

_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|related work)\s*$", re.IGNORECASE | re.MULTILINE
)


def locate_references_page(doc: pymupdf.Document, page_texts: list[str]) -> int | None:
    """Find the page a paper's reference section starts on, if possible.

    Args:
        doc (pymupdf.Document): The open PDF document, used to check its outline/bookmarks.
        page_texts (list[str]): Each page's extracted text, in page order.

    Returns:
        int | None: The 0-based page index the reference section starts on, or None
            if it couldn't be located via either the outline or a heading scan.
    """
    for level, title, page_number in doc.get_toc():
        if title.strip().lower() in ("references", "bibliography", "related work"):
            return page_number - 1

    for page_index in range(len(page_texts) - 1, -1, -1):
        if _HEADING_RE.search(page_texts[page_index]):
            return page_index

    return None


def split_body_and_references(pdf_bytes: bytes) -> tuple[str, str | None]:
    """Split a PDF's text into body text and reference-section text.

    Args:
        pdf_bytes (bytes): The raw bytes of a PDF file.

    Returns:
        tuple[str, str | None]: (body_text, references_text). references_text is
            None if no reference section could be located, in which case body_text
            is the paper's full text.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_texts = [str(page.get_text("text")) for page in doc]
        start_page = locate_references_page(doc, page_texts)
    finally:
        doc.close()

    if start_page is None:
        return "\n\n".join(page_texts).replace("\x00", ""), None

    body_text = "\n\n".join(page_texts[:start_page]).replace("\x00", "")
    references_text = "\n\n".join(page_texts[start_page:]).replace("\x00", "")
    return body_text, references_text


def embed_in_batches(texts: list[str], batch_size: int = MAX_EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed texts in batches, since the embedding server caps how many it accepts per call.

    Args:
        texts (list[str]): The texts to embed.
        batch_size (int, optional): Max texts per embedding call. Defaults to 32,
            matching the embedding server's maximum batch size.

    Returns:
        list[list[float]]: The embedding vectors, in the same order as `texts`.
    """
    vectors = []
    for i in range(0, len(texts), batch_size):
        vectors.extend(embed_texts(texts[i : i + batch_size]))
    return vectors


def filter_bibliography_chunks(
    pdf_bytes: bytes, chunks: list[str], threshold: float = SIMILARITY_THRESHOLD
) -> list[str]:
    """Drop chunks that closely match a paper's own reference-section text.

    Args:
        pdf_bytes (bytes): The raw bytes of the PDF the chunks were produced from.
        chunks (list[str]): The candidate chunks to filter.
        threshold (float, optional): Cosine-similarity cutoff above which a chunk is
            considered bibliography content. Defaults to config's similarity_threshold.

    Returns:
        list[str]: The chunks that aren't close matches to the reference section, in
            their original order. Returns `chunks` unchanged if filtering is disabled
            or no reference section could be located.
    """
    _, references_text = split_body_and_references(pdf_bytes)
    if not references_text:
        return chunks

    reference_fragments = recursive_chunk_text(
        references_text, chunk_size=300, chunk_overlap=0
    )
    if not reference_fragments:
        return chunks

    chunk_vectors = np.array(embed_in_batches(chunks))
    reference_vectors = np.array(embed_in_batches(reference_fragments))

    chunk_norms = chunk_vectors / np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    reference_norms = reference_vectors / np.linalg.norm(
        reference_vectors, axis=1, keepdims=True
    )
    similarity = chunk_norms @ reference_norms.T
    max_similarity = similarity.max(axis=1)

    return [chunk for chunk, score in zip(chunks, max_similarity) if score <= threshold]
