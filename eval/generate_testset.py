"""Generate a single-hop/multi-hop QA testset by calling an LLM directly.

Single-hop samples come from one paper's full text; multi-hop samples come from a
chunk and its cross-paper neighbors in the `edges` table. No RAGAS TestsetGenerator
involved — this is a lighter, custom alternative.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel
from sqlalchemy import func, select

from chunking.chunker import extract_pdf_text
from clients.llm_client import chat_llm
from corpus.fetch_corpus import _is_pdf_reachable
from store.db import SessionLocal, init_db
from store.models import Chunk, Edge, PaperRecord
from eval.prompts import MULTI_HOP_SYSTEM_PROMPT, SINGLE_HOP_SYSTEM_PROMPT


class SingleHopSample(BaseModel):
    question: str
    answer: str
    excerpt: str


class SingleHopResponse(BaseModel):
    samples: list[SingleHopSample]


class MultiHopResult(BaseModel):
    answerable_from_single_excerpt: bool
    question: str | None = None
    answer: str | None = None


def get_random_unused_paper(used_ids: set[int]) -> PaperRecord | None:
    """Pick a random paper not already in `used_ids`.

    Args:
        used_ids (set[int]): Paper ids to exclude, so a paper is never sampled twice.

    Returns:
        PaperRecord | None: A random unused paper, or None if none remain.
    """
    with SessionLocal() as session:
        stmt = select(PaperRecord).order_by(func.random())
        if used_ids:
            stmt = stmt.where(PaperRecord.id.not_in(used_ids))
        return session.execute(stmt.limit(1)).scalars().first()


def fetch_paper_text(
    paper: PaperRecord, session_: requests.Session, timeout: float = 10.0
) -> str | None:
    """Download a paper's PDF and extract its full text, if reachable.

    Args:
        paper (PaperRecord): The paper whose PDF should be fetched.
        session_ (requests.Session): Shared HTTP session for connection reuse.
        timeout (float, optional): Seconds to wait for reachability/download. Defaults to 10.0.

    Returns:
        str | None: The paper's full extracted text, or None if the PDF isn't reachable.
    """
    if not _is_pdf_reachable(paper.pdf_url, session_, timeout):
        return None
    pdf_bytes = session_.get(paper.pdf_url, timeout=timeout).content
    return extract_pdf_text(pdf_bytes)


def generate_single_hop_samples(count: int) -> list[dict[str, Any]]:
    """Generate single-hop QA samples, each drawn from one paper not reused across samples.

    Args:
        count (int): How many single-hop samples to generate.

    Returns:
        list[dict[str, Any]]: Up to `count` single-hop samples.
    """
    samples: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    session_ = requests.Session()
    structured_llm = chat_llm.with_structured_output(SingleHopResponse)

    while len(samples) < count:
        paper = get_random_unused_paper(used_ids)
        if paper is None:
            print(
                f"No unused papers remain; stopping with {len(samples)}/{count} single-hop samples."
            )
            break

        used_ids.add(paper.id)
        text = fetch_paper_text(paper, session_)
        if text is None:
            continue

        try:
            response = structured_llm.invoke(
                [
                    {"role": "system", "content": SINGLE_HOP_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ]
            )
            for sample in response.samples:
                samples.append(
                    {
                        "group": "single_hop",
                        "question": sample.question,
                        "answer": sample.answer,
                        "reference_contexts": [sample.excerpt],
                        "arxiv_ids": [paper.arxiv_id],
                    }
                )
            print(f"Single-hop: {len(samples)}/{count} (paper {paper.arxiv_id})")
        except Exception as e:
            print(e)
            break

    return samples[:count]


def get_random_cross_paper_edge() -> tuple[Chunk, list[Chunk]] | None:
    """Pick a random edge row and return its source chunk plus cross-paper target chunks.

    Args:
        None

    Returns:
        tuple[Chunk, list[Chunk]] | None: (source chunk, 2-3 target chunks from a
            different paper), or None if the sampled edge has fewer than 2 such targets.
    """
    with SessionLocal() as session:
        edge = (
            session.execute(select(Edge).order_by(func.random()).limit(1))
            .scalars()
            .first()
        )
        if edge is None:
            return None

        source = session.get(Chunk, edge.source_chunk_id)
        if source is None:
            return None

        targets = (
            session.execute(select(Chunk).where(Chunk.id.in_(edge.target_chunk_ids)))
            .scalars()
            .all()
        )
        cross_paper_targets = [
            chunk for chunk in targets if chunk.paper_id != source.paper_id
        ]
        if len(cross_paper_targets) < 2:
            return None

        chosen = random.sample(cross_paper_targets, k=min(2, len(cross_paper_targets)))
        return source, chosen


def generate_multi_hop_samples(
    count: int, max_attempts: int | None = None
) -> list[dict[str, Any]]:
    """Generate multi-hop QA samples from cross-paper chunk excerpts.

    Args:
        count (int): How many multi-hop samples to generate.
        max_attempts (int | None, optional): Cap on edge-sampling attempts before giving
            up early. Defaults to None, which uses `count * 10`.

    Returns:
        list[dict[str, Any]]: Up to `count` multi-hop samples.
    """
    if max_attempts is None:
        max_attempts = count * 10

    samples: list[dict[str, Any]] = []
    structured_llm = chat_llm.with_structured_output(MultiHopResult)

    attempts = 0
    while len(samples) < count and attempts < max_attempts:
        attempts += 1
        edge = get_random_cross_paper_edge()
        if edge is None:
            continue
        source, targets = edge
        all_chunks = [source] + targets

        with SessionLocal() as session:
            papers = (
                session.execute(
                    select(PaperRecord).where(
                        PaperRecord.id.in_({chunk.paper_id for chunk in all_chunks})
                    )
                )
                .scalars()
                .all()
            )
        papers_by_id = {paper.id: paper for paper in papers}

        labels = string.ascii_uppercase[: len(all_chunks)]
        excerpts_text = "\n\n---\n\n".join(
            f'Excerpt {label} (Paper: "{papers_by_id[chunk.paper_id].title}"):\n{chunk.content}'
            for label, chunk in zip(labels, all_chunks)
        )

        try:
            result = structured_llm.invoke(
                [
                    {"role": "system", "content": MULTI_HOP_SYSTEM_PROMPT},
                    {"role": "user", "content": excerpts_text},
                ]
            )
        except Exception as e:
            print(e)
            break

        if (
            result.answerable_from_single_excerpt
            or not result.question
            or not result.answer
        ):
            continue

        samples.append(
            {
                "group": "multi_hop",
                "question": result.question,
                "answer": result.answer,
                "reference_contexts": [chunk.content for chunk in all_chunks],
                "arxiv_ids": [
                    papers_by_id[chunk.paper_id].arxiv_id for chunk in all_chunks
                ],
            }
        )
        print(f"Multi-hop: {len(samples)}/{count} (attempt {attempts}/{max_attempts})")

    if len(samples) < count:
        print(
            f"Stopped after {attempts} attempts with {len(samples)}/{count} multi-hop samples."
        )

    return samples


def save_samples_to_jsonl(samples: list[dict[str, Any]], path: str | Path) -> None:
    """Write samples to a JSONL file, one JSON object per line.

    Args:
        samples (list[dict[str, Any]]): The samples to write.
        path (str | Path): Destination file path.

    Returns:
        None
    """
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def run_single_hop_generation(count: int) -> None:
    """Generate single-hop samples and save them to a JSONL file.

    Args:
        count (int): How many single-hop samples to generate.

    Returns:
        None
    """
    OUTPUT_PATH = Path("tests/datasets/single_hop_samples.jsonl")
    samples = generate_single_hop_samples(count)
    save_samples_to_jsonl(samples, OUTPUT_PATH)
    print(f"Saved {len(samples)} single-hop samples to {OUTPUT_PATH}")


def run_multi_hop_generation(count: int) -> None:
    """Generate multi-hop samples and save them to a JSONL file.

    Args:
        count (int): How many multi-hop samples to generate.

    Returns:
        None
    """
    OUTPUT_PATH = Path("tests/datasets/multi_hop_samples.jsonl")
    samples = generate_multi_hop_samples(count)
    save_samples_to_jsonl(samples, OUTPUT_PATH)
    print(f"Saved {len(samples)} multi-hop samples to {OUTPUT_PATH}")


if __name__ == "__main__":

    init_db()
    # run_single_hop_generation(count=15)
    run_multi_hop_generation(count=15)
