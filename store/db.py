"""Database engine/session setup and persistence for fetched papers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

from corpus.schemas import PAPERS_ADAPTER
from store.models import Base, PaperRecord

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"])
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(engine)


def save_papers_to_db(json_path: str | Path) -> int:
    papers = PAPERS_ADAPTER.validate_json(Path(json_path).read_bytes())
    rows = [paper.model_dump() for paper in papers]
    if not rows:
        return 0

    stmt = (
        insert(PaperRecord)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["arxiv_id"])
        .returning(PaperRecord.id)
    )

    with SessionLocal() as session:
        result = session.execute(stmt)
        session.commit()
        return len(result.fetchall())
