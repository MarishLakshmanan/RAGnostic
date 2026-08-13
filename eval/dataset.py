"""Extract each paper's full PDF text from Postgres as a RAGAS-ready Document."""

from __future__ import annotations

from typing import Iterator

import requests
from langchain_core.documents import Document
from sqlalchemy import select

from chunking.chunker import extract_pdf_text
from corpus.fetch_corpus import _is_pdf_reachable
from store.db import SessionLocal
from store.models import PaperRecord


def iter_paper_documents(timeout: float = 10.0) -> Iterator[Document]:
    """Yield a Document per paper in the DB, skipping papers whose PDF isn't reachable.

    Args:
        timeout (float, optional): How many seconds to wait for the PDF reachability
            check and download before giving up. Defaults to 10.0.

    Yields:
        Document: One LangChain Document per paper, holding the paper's full extracted
            text as page_content and arxiv_id/title as metadata.
    """
    session_ = requests.Session()

    with SessionLocal() as session:
        papers = session.execute(select(PaperRecord)).scalars()
        for paper in papers:
            if not _is_pdf_reachable(paper.pdf_url, session_, timeout):
                continue
            pdf_bytes = session_.get(paper.pdf_url, timeout=timeout).content
            text = extract_pdf_text(pdf_bytes)
            yield Document(
                page_content=text,
                metadata={"arxiv_id": paper.arxiv_id, "title": paper.title},
            )
