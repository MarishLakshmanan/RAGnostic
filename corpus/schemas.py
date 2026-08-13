"""Pydantic data models for the corpus pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, TypeAdapter, ConfigDict


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime
    pdf_url: str


PAPERS_ADAPTER = TypeAdapter(list[Paper])


class ChunkModel(BaseModel):
    id: int
    paper_id: int
    chunk_index: int
    content: str
    strategy: str
    embedded_at: datetime | None

    model_config = ConfigDict(from_attributes=True)
