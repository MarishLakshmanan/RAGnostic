"""Shared fixtures for RAGAS-based quality tests."""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from datetime import datetime, timezone

import pytest
from ragas.embeddings import BaseRagasEmbedding, BaseRagasEmbeddings, embedding_factory
from ragas.llms import InstructorBaseRagasLLM, llm_factory

from clients.embedding_client import (
    EMBEDDING_MODEL,
    async_client as embedding_openai_client,
)
from clients.llm_client import LLM_MODEL, async_client as llm_openai_client

TESTSET_PATH = Path(__file__).resolve().parent / "datasets"


def _load_testset_group(group: str) -> list[dict]:
    """Load testset rows belonging to one sample group.

    Args:
        group (str): Which group to load, "single_hop" or "multi_hop".

    Returns:
        list[dict]: The matching rows from eval/testset.jsonl.
    """
    rows = []
    file_name = (
        "multi_hop_samples.jsonl"
        if group == "multi_hop"
        else "single_hop_samples.jsonl"
    )
    file = TESTSET_PATH / file_name
    with open(file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    return rows


def pytest_addoption(parser):
    parser.addoption("--changes", action="store")
    parser.addoption("--name", action="store")


@pytest.fixture
def single_hop_rows() -> list[dict]:
    return _load_testset_group("single_hop")


@pytest.fixture
def multi_hop_rows() -> list[dict]:
    return _load_testset_group("multi_hop")


@pytest.fixture(scope="session")
def ragas_llm() -> InstructorBaseRagasLLM:
    return llm_factory(
        LLM_MODEL,
        provider="openai",
        client=llm_openai_client,
    )


@pytest.fixture(scope="session")
def ragas_embeddings() -> BaseRagasEmbeddings | BaseRagasEmbedding:
    return embedding_factory(
        provider="openai",
        model=EMBEDDING_MODEL,
        client=embedding_openai_client,
    )


@pytest.fixture(scope="session")
def results_dir(request) -> Path:

    changes = request.config.getoption("--changes")
    name = request.config.getoption("--name")

    if not changes and not name:
        raise ValueError("Either --changes or --name must be provided")

    config_file = Path("./config.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)["config"]
    file_name = (
        config["chunking_strategy"]["name"]
        + "_"
        + config["retrieval_strategy"]["name"]
        + "_"
        + config["reranking_strategy"]["name"]
    )
    path = Path(__file__).resolve().parent / "results" / file_name
    path.mkdir(parents=True, exist_ok=True)

    config["timestamp"] = datetime.now(timezone.utc)
    config["name"] = name
    config["changes"] = changes

    with open((path / "config.yaml"), "w") as f:
        yaml.dump(config, f)

    return path
