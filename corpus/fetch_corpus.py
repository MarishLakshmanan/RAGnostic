"""Fetch arXiv papers matching a query, yielding only ones with a reachable pdf_url.

Streams results via a generator: if some fraction of a batch's PDFs turn out to be
unreachable, transparently fetches more pages from arXiv until `count` reachable
papers have been yielded (or arXiv's results are exhausted).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import requests

from corpus.schemas import PAPERS_ADAPTER, Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "rag-eval-corpus-fetcher/0.1 (mailto:sakthi9481@gmail.com)"
ARXIV_RATE_LIMIT_SECONDS = 3.0


def _parse_atom_feed(xml_bytes: bytes) -> list[Paper]:
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = (entry.findtext(f"{ATOM_NS}id") or "").strip()
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
        summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip()
        published_text = (entry.findtext(f"{ATOM_NS}published") or "").strip()
        authors = [
            (name.text or "").strip()
            for name in entry.findall(f"{ATOM_NS}author/{ATOM_NS}name")
        ]

        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        if not pdf_url or not arxiv_id or not published_text:
            continue

        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                summary=summary,
                published=datetime.fromisoformat(published_text),
                pdf_url=pdf_url,
            )
        )
    return papers


def _fetch_page(
    query: str,
    start: int,
    max_results: int,
    session: requests.Session,
    timeout: float,
    max_retries: int = 3,
) -> list[Paper]:
    last_error: requests.RequestException = requests.RequestException(
        "max_retries must be >= 1"
    )
    for attempt in range(max_retries):
        try:
            response = session.get(
                ARXIV_API_URL,
                params={
                    "search_query": query,
                    "start": start,
                    "max_results": max_results,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return _parse_atom_feed(response.content)
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise last_error


def _is_pdf_reachable(url: str, session: requests.Session, timeout: float) -> bool:
    try:
        response = session.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code == 405:
            response = session.get(url, stream=True, timeout=timeout)
            response.close()
        return response.status_code < 400
    except requests.RequestException:
        return False


def fetch_corpus(
    query: str, count: int, page_size: int = 50, timeout: float = 10.0
) -> Iterator[Paper]:
    """Yield up to `count` arXiv papers matching `query` whose pdf_url is reachable.

    Fetches additional pages from the arXiv API as needed to backfill papers whose
    PDFs turn out to be unreachable, stopping once `count` reachable papers have
    been yielded or arXiv's results are exhausted.
    """
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    start = 0
    yielded = 0
    first_request = True

    while yielded < count:
        if not first_request:
            time.sleep(ARXIV_RATE_LIMIT_SECONDS)
        first_request = False

        page = _fetch_page(query, start, page_size, session, timeout)
        if not page:
            break
        start += len(page)

        for paper in page:
            if _is_pdf_reachable(paper.pdf_url, session, timeout):
                yield paper
                yielded += 1
                if yielded >= count:
                    return


def save_papers_to_json(papers: Iterable[Paper], path: str | Path) -> None:
    Path(path).write_bytes(PAPERS_ADAPTER.dump_json(list(papers), indent=2))


if __name__ == "__main__":
    from store.db import init_db, save_papers_to_db

    QUERY = (
        '(ti:"LLM security" OR abs:"LLM security") OR '
        '(ti:"securing LLMs" OR abs:"securing LLMs") OR '
        '(ti:"LLM safety" OR abs:"LLM safety") OR '
        '(ti:"language model security" OR abs:"language model security") OR '
        '(ti:"AI agent security" OR abs:"AI agent security") OR '
        '(ti:"agentic AI security" OR abs:"agentic AI security") OR '
        '(ti:"securing AI agents" OR abs:"securing AI agents") OR '
        '(ti:"LLM vulnerabilities" OR abs:"LLM vulnerabilities") OR '
        '(ti:"LLM attacks" OR abs:"LLM attacks") OR '
        '(ti:"adversarial attacks on LLMs" OR abs:"adversarial attacks on LLMs") OR '
        '(ti:"prompt injection" OR abs:"prompt injection") OR '
        '(ti:"jailbreak" OR abs:"jailbreak") OR '
        '(ti:"jailbreaking LLMs" OR abs:"jailbreaking LLMs") OR '
        '(ti:"red teaming LLMs" OR abs:"red teaming LLMs") OR '
        '(ti:"red teaming language models" OR abs:"red teaming language models") OR '
        '(ti:"AI agent safety" OR abs:"AI agent safety") OR '
        '(ti:"autonomous agent security" OR abs:"autonomous agent security") OR '
        '(ti:"LLM robustness" OR abs:"LLM robustness") OR '
        '(ti:"trustworthy LLM" OR abs:"trustworthy LLM") OR '
        '(ti:"secure AI agents" OR abs:"secure AI agents") OR '
        '(ti:"cybersecurity of large language models" OR abs:"cybersecurity of large language models") OR '
        '(ti:"LLM threat model" OR abs:"LLM threat model") OR '
        '(ti:"defending LLMs" OR abs:"defending LLMs")'
    )

    init_db()

    collected: list[Paper] = []
    for i, paper in enumerate(fetch_corpus(QUERY, count=50), start=1):
        print(f"{i:>3}. [{paper.arxiv_id}] {paper.title}")
        print(f"     {paper.pdf_url}")
        collected.append(paper)

    output_path = Path(__file__).parent / "papers.json"
    save_papers_to_json(collected, output_path)
    print(f"\nSaved {len(collected)} papers to {output_path}")

    inserted = save_papers_to_db(output_path)
    print(f"Inserted {inserted} new ({len(collected) - inserted} already existed).")
