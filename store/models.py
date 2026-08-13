"""SQLAlchemy ORM models for the corpus pipeline."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PaperRecord(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(
        String, unique=True, nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    authors: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    published: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    pdf_url: Mapped[str] = mapped_column(String, nullable=False)


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (
        UniqueConstraint(
            "paper_id",
            "chunk_index",
            "strategy",
            name="uq_chunk_paper_id_chunk_index_strategy",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Edge(Base):
    __tablename__ = "edges"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunk.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    target_chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
