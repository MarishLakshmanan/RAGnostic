"""RAG pipeline entrypoint: embed a question, retrieve chunks, answer via the LLM."""

from __future__ import annotations

from typing import Any

import bm25s
import yaml
from sqlalchemy import select

from clients.bm25_client import get_bm25_retriever
from clients.chroma_client import get_collection
from clients.embedding_client import embed_texts
from clients.llm_client import chat_llm
from clients.reranker_client import rerank
from store.db import SessionLocal
from store.models import Chunk, Edge, PaperRecord
from langfuse import observe
from corpus.schemas import ChunkModel

SYSTEM_PROMPT_TEMPLATE = """You are a research assistant answering questions about a corpus of \
LLM/agent-security papers. Use only the context below to answer the user's question. If the \
context doesn't contain the answer, say so instead of guessing.

Context:
{context}"""

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["config"]
    RETRIEVAL_STRATEGY = config["retrieval_strategy"]["name"]
    RERANKING_STRATEGY = config["reranking_strategy"]["name"]


@observe()
def retrieve_top_chunks(query: str, top_k: int = 5) -> list[ChunkModel]:
    """Embed a query and fetch its top-k nearest chunks from Chroma, in ranked order.

    Args:
        query (str): The question to retrieve context for.
        top_k (int, optional): How many chunks to retrieve. Defaults to 5.

    Returns:
        list[ChunkModel]: The top-k chunks, ordered from most to least similar.
    """
    vector = embed_texts([query])[0]
    result = get_collection().query(query_embeddings=[vector], n_results=top_k)
    ranked_ids = [int(chunk_id) for chunk_id in result["ids"][0]]

    with SessionLocal() as session:
        chunks = (
            session.execute(select(Chunk).where(Chunk.id.in_(ranked_ids)))
            .scalars()
            .all()
        )

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    retrieved_chunks = [
        chunks_by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in chunks_by_id
    ]

    # Convert SQLAlchemy Chunk objects to Pydantic ChunkModel objects for langfuse since it can't serialize SQLAlchemy objects directly.
    chunks_models = [ChunkModel.model_validate(chunk) for chunk in retrieved_chunks]
    return chunks_models


@observe()
def retrieve_bm25_chunks(query: str, top_k: int = 5) -> list[ChunkModel]:
    """Fetch a query's top-k nearest chunks via BM25 sparse (keyword) search, in ranked order.

    Args:
        query (str): The question to retrieve context for.
        top_k (int, optional): How many chunks to retrieve. Defaults to 5.

    Returns:
        list[ChunkModel]: The top-k chunks, ordered from most to least relevant.
    """
    tokens = bm25s.tokenize([query], stopwords="en")
    results, _scores = get_bm25_retriever().retrieve(tokens, k=top_k)
    ranked_ids = [int(row["id"]) for row in results[0]]

    with SessionLocal() as session:
        chunks = (
            session.execute(select(Chunk).where(Chunk.id.in_(ranked_ids)))
            .scalars()
            .all()
        )

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    retrieved_chunks = [
        chunks_by_id[chunk_id] for chunk_id in ranked_ids if chunk_id in chunks_by_id
    ]
    return [ChunkModel.model_validate(chunk) for chunk in retrieved_chunks]


@observe()
def reciprocal_rank_fusion(
    ranked_lists: list[list[ChunkModel]], top_k: int = 5, k: int = 60
) -> list[ChunkModel]:
    """Merge multiple ranked chunk lists into one via Reciprocal Rank Fusion.

    Args:
        ranked_lists (list[list[ChunkModel]]): Ranked chunk lists to merge, e.g. dense and BM25 results.
        top_k (int, optional): How many chunks to keep after merging. Defaults to 5.
        k (int, optional): RRF's rank-dampening constant. Defaults to 60.

    Returns:
        list[ChunkModel]: The merged, deduplicated top-k chunks, ordered by fused score.
    """
    scores: dict[int, float] = {}
    chunks_by_id: dict[int, ChunkModel] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id.setdefault(chunk.id, chunk)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks_by_id[chunk_id] for chunk_id in ranked_ids[:top_k]]


@observe()
def expand_with_graph(chunk: Chunk) -> list[Chunk]:
    """Expand a chunk with its nearest-neighbor chunks from the `edges` table.

    Args:
        chunk (Chunk): The chunk to expand.

    Returns:
        list[Chunk]: `chunk` followed by its stored graph neighbors, or just `[chunk]`
            if it has no edge row.
    """
    with SessionLocal() as session:
        edge = (
            session.execute(select(Edge).where(Edge.source_chunk_id == chunk.id))
            .scalars()
            .first()
        )
        if edge is None:
            return [chunk]

        neighbors = (
            session.execute(select(Chunk).where(Chunk.id.in_(edge.target_chunk_ids)))
            .scalars()
            .all()
        )

    return [chunk] + list(neighbors)


@observe()
def rerank_chunks(query: str, chunks: list[ChunkModel], top_k: int = 5) -> list[ChunkModel]:
    """Rerank chunks against the query using the cross-encoder and keep the top-k.

    Args:
        query (str): The question to score chunks against.
        chunks (list[ChunkModel]): The candidate chunks to rerank.
        top_k (int, optional): How many chunks to keep after reranking. Defaults to 5.

    Returns:
        list[ChunkModel]: The top-k chunks, ordered by cross-encoder score descending.
    """
    results = rerank(query, [chunk.content for chunk in chunks])
    return [chunks[result["index"]] for result in results[:top_k]]


@observe()
def build_context(chunks: list[ChunkModel]) -> str:
    """Join chunk texts into a single context string.

    Args:
        chunks (list[ChunkModel]): The chunks to include as context, in order.

    Returns:
        str: The chunk texts joined with a "---" delimiter.
    """
    return "\n\n---\n\n".join(chunk.content for chunk in chunks)


@observe()
def get_arxiv_ids_for_chunks(chunks: list[ChunkModel]) -> list[str]:
    """Resolve retrieved chunks to their source papers' arxiv ids.

    Args:
        chunks (list[ChunkModel]): The retrieved chunks to resolve.

    Returns:
        list[str]: The arxiv id of each chunk's source paper.
    """
    paper_ids = {chunk.paper_id for chunk in chunks}
    with SessionLocal() as session:
        rows = session.execute(
            select(PaperRecord.id, PaperRecord.arxiv_id).where(
                PaperRecord.id.in_(paper_ids)
            )
        ).all()

    arxiv_id_by_paper_id = {row.id: row.arxiv_id for row in rows}
    return [arxiv_id_by_paper_id[chunk.paper_id] for chunk in chunks]


@observe()
def answer_question(
    question: str, use_graph: bool = False, top_k: int = 5
) -> dict[str, Any]:
    """Answer a question by retrieving context and generating a response with the LLM.

    Args:
        question (str): The question to answer.
        use_graph (bool, optional): If True, expand only the single best-matching chunk
            using its `edges` neighbors instead of using the full top-k vector search
            results, overriding `config.yaml`'s `retrieval_strategy`. Defaults to False.
        top_k (int, optional): How many chunks to retrieve via vector search. Defaults to 5.

    Returns:
        dict[str, Any]: {"answer": str, "contexts": list[str], "arxiv_ids": list[str]} —
            the generated answer, the chunk texts used as context, and the arxiv ids of
            the papers those chunks came from.
    """
    if use_graph or RETRIEVAL_STRATEGY == "graph":
        chunks = retrieve_top_chunks(question, top_k)
        chunks = expand_with_graph(chunks[0])
    elif RETRIEVAL_STRATEGY == "hybrid":
        dense_chunks = retrieve_top_chunks(question, top_k)
        sparse_chunks = retrieve_bm25_chunks(question, top_k)
        chunks = reciprocal_rank_fusion([dense_chunks, sparse_chunks], top_k)
    else:
        chunks = retrieve_top_chunks(question, top_k)

    if RERANKING_STRATEGY == "cross_encoder":
        chunks = rerank_chunks(question, chunks, top_k)

    context = build_context(chunks)
    response = chat_llm.invoke(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(context=context),
            },
            {"role": "user", "content": question},
        ]
    )

    return {
        "answer": response.content,
        "contexts": [chunk.content for chunk in chunks],
        "arxiv_ids": get_arxiv_ids_for_chunks(chunks),
    }


if __name__ == "__main__":
    result = answer_question("What is prompt injection?")
    print(result["answer"])
    print(f"\n({len(result['contexts'])} context chunks used)")
