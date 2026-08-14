# RAGnostic

An eval harness for comparing RAG retrieval strategies dense, graph, and hybrid, with an optional cross-encoder reranker over a self-fetched corpus of LLM/agent-security arXiv papers. Scored with RAGAS, running entirely on local, open-weight models.

The repo is split by pipeline stage, where each step acts as a checkpoint: state (Postgres rows, Chroma vectors, on-disk indexes) is upserted and marked as it's produced, so re-running any stage is a safe, deduplicated no-op rather than a redo from scratch.

**Everything runs locally.** No external API calls, no API keys. The LLM is served by vLLM, and both the embedding model and the cross-encoder reranker are served by Hugging Face's Text Embeddings Inference (TEI), all three as Docker containers defined in `infra/`.

## Project Structure

- `corpus/` fetches papers from the arXiv API and loads them into Postgres.
- `store/` the SQLAlchemy models and DB session/engine setup shared by every stage.
- `chunking/` downloads each paper's PDF, strips out bibliography/reference-list text, and chunks the body for retrieval.
- `embed/` embeds chunks into ChromaDB, and builds a BM25 sparse index for hybrid search.
- `graph/` builds a chunk-adjacency graph from Chroma nearest-neighbors, for Graph RAG expansion.
- `clients/` thin HTTP clients for the embedding server, Chroma, the chat LLM, and the BM25 index.
- `eval/` generates a synthetic single-hop/multi-hop QA testset by prompting an LLM directly over the corpus.
- `tests/` RAGAS-based quality gates that run the pipeline end-to-end and score it against the testset.
- `web/` renders eval results into a static HTML report for browsing runs.
- `infra/` Docker Compose + service configs for Postgres, ChromaDB, the embedding model, the LLM, and the cross-encoder.
- `main.py` the query-side entrypoint: takes a question, retrieves context, and generates an answer.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/), Python 3.13+, and Docker.

1. Bring up the local services (Postgres, ChromaDB, embedding model, LLM, cross-encoder) with `docker compose -f infra/docker-compose.yaml up -d`.
2. Create a `.env` with `DATABASE_URL`, `CHROMA_HOST`/`CHROMA_PORT`/`CHROMA_COLLECTION`, `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL`, and `LLM_BASE_URL`/`LLM_MODEL`. None of the local services need an API key.
3. `uv sync` to install dependencies.

## The Pipeline

The corpus is generated using `uv run python -m corpus.fetch_corpus`. The fetched papers get stored in the DB and also written out as a json file.

Then we chunk them using `uv run python -m chunking.parser`. This filters the bibliography text out of the pdf text using a vector embedding approach, then chunks the actual content and stores it in the DB.

Then we embed the chunks using `uv run python -m embed.embed`. This updates Postgres to mark that the chunk's been embedded (so we don't redo it) and pushes the vector into ChromaDB.

Optionally, we can also run `uv run python -m embed.build_bm25_index` to build an index for BM25, needed for the `hybrid` retrieval strategy, which merges dense and BM25 results with reciprocal rank fusion (RRF).

## Configuration

`config.yaml` drives the tunable parts of the pipeline:

- `retrieval_strategy.name` `dense`, `graph`, `hybrid`, or `none`. Read by `main.py` to decide how a question gets answered.
- `chunking_strategy.name` which chunking function `chunking/parser.py` runs PDFs through.
- `reranking_strategy.name` `cross_encoder` or `none`. Currently declared and used for labeling eval runs, but not yet wired into the retrieval path itself.
- `bibliography_filter.similarity_threshold` how aggressively `chunking/bibliography_filter.py` drops reference-list chunks. Uses max (not average) cosine similarity against reference fragments, so a chunk is dropped if it closely matches even a single citation.

## Graph RAG

But there's more:

We can turn this into Graph RAG by building a knowledge graph with `uv run python -m graph.build_graph`. This queries Chroma for each embedded chunk's nearest neighbors and stores the adjacency in Postgres, which `main.py` can then expand outward from during retrieval.

## Evaluation

Then we evaluate everything using RAGAS.

First we generate the test dataset using `uv run python -m eval.generate_testset`, then just run `uv run pytest`.

Each test scores answers on faithfulness, answer relevancy, and a paper-level recall metric (whether the retrieved chunks came from the testset row's ground-truth source papers). Results get stored as per-row CSV files inside `tests/results/`.

You can view them in your browser by running `uv run python -m web.build`. This builds a static html site using jinja2 templates.

You can check out my tests and results here: [Results](https://marishlakshmanan.github.io/RAGnostic/)

## Status

Dense, graph, and hybrid retrieval along with cross encoder are implemented and swappable via `config.yaml`. Chunking currently has one strategy (`recursive_chunker`), designed to be swappable without touching the rest of the pipeline.
