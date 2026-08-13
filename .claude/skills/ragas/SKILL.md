---
name: ragas
description: Use this skill when working with RAGAS (ragas-io) — generating synthetic RAG test datasets via its Knowledge Graph pipeline (SingleHopSpecific/MultiHopSpecific/SingleHopAbstract/MultiHopAbstract query synthesizers), or evaluating a RAG pipeline with ragas.evaluate() and its metrics (faithfulness, context_precision, context_recall, answer_relevancy, etc). Covers core concepts, dataset generation, evaluation, and wiring RAGAS to self-hosted OpenAI-compatible LLM/embedding endpoints like this project's.
---

# RAGAS

RAGAS (`ragas` on PyPI) is a framework for two related jobs: **generating synthetic test datasets** for a RAG pipeline, and **evaluating** a RAG pipeline's outputs against metrics. They share vocabulary but are separate workflows.

## Mental model

**Testset generation**: `KnowledgeGraph` (built from your documents) → `transforms` enrich it (NER, keyphrases, relationship-building between chunks) → `Persona` objects describe likely user types → a `query_distribution` of `QuerySynthesizer`s draws scenarios from the graph → `TestsetGenerator.generate(...)` turns scenarios into question/context/reference samples.

**Evaluation**: an `EvaluationDataset` of `SingleTurnSample`s (`user_input`, `retrieved_contexts`, `response`, `reference`) → `ragas.evaluate(dataset, metrics=[...])` scores each sample against one or more metrics, most of which are themselves LLM judges under the hood.

Both need an `llm` and/or `embedding_model` — RAGAS never calls a model provider directly, it wraps whatever LLM/embeddings object you hand it (see `references/custom_models.md` for wiring up a self-hosted OpenAI-compatible endpoint like this project's).

## Quick start — generate a testset

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)  # docs: list[langchain_core.documents.Document]
```

This picks up RAGAS's default query distribution (SingleHopSpecific/MultiHopAbstract/MultiHopSpecific only — no SingleHopAbstract). For full control over the knowledge graph and to include all four query types, see `references/testset_generation.md`.

## Quick start — evaluate

```python
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision

dataset = EvaluationDataset(samples=[
    SingleTurnSample(
        user_input="What is Ragas?",
        retrieved_contexts=["Ragas is an evaluation framework for LLM applications."],
        response="Ragas is a framework for evaluating RAG pipelines.",
        reference="Ragas is an evaluation framework for LLM applications.",
    ),
])

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
```

See `references/evaluation.md` for the full metrics catalog and which metrics need `reference` (ground truth) vs. are reference-free.

## The four query synthesizer types

| Synthesizer | Hops | Style |
|---|---|---|
| `SingleHopSpecificQuerySynthesizer` | 1 node | Direct, fact-based question answerable from one chunk |
| `SingleHopAbstractQuerySynthesizer` | 1 node | Interpretive/conceptual question about one chunk, not a single fact lookup |
| `MultiHopSpecificQuerySynthesizer` | 2+ related nodes | Fact-based question whose answer requires combining specific facts from multiple chunks |
| `MultiHopAbstractQuerySynthesizer` | 2+ related nodes | Broad/thematic question synthesizing ideas across multiple chunks |

Details, imports, and how to build an explicit `query_distribution` covering all four: `references/testset_generation.md`.

## References

- `references/testset_generation.md` — KnowledgeGraph construction, transforms, personas, query synthesizers, building a custom `query_distribution`, output schema.
- `references/evaluation.md` — `SingleTurnSample`/`EvaluationDataset` schema, the metrics catalog, `evaluate()`, custom metrics.
- `references/custom_models.md` — wrapping a self-hosted OpenAI-compatible LLM/embedding endpoint for RAGAS.

## Using RAGAS in this project (RAG-eval)

The `eval/` stage (see `plan.md`) generates a testset from the papers already loaded in Postgres, then evaluates the RAG pipeline against it. Reuse existing code rather than re-implementing:

- **Iterating papers**: reuse the `SessionLocal` + `PaperRecord` pattern from [`chunking/parser.py`](../../../chunking/parser.py)'s `iter_paper_chunks` to `SELECT * FROM papers`.
- **PDF → text**: reuse `corpus.fetch_corpus._is_pdf_reachable` for the reachability check (same as `iter_paper_chunks` does), then extract text the same way [`chunking/chunker.py`](../../../chunking/chunker.py)'s `chunk_pdf_bytes` does — `pymupdf.open(stream=pdf_bytes, filetype="pdf")`, join `page.get_text("text")` per page, strip `\x00`. **Don't** call `chunk_pdf_bytes` itself for this — RAGAS wants each paper as one whole `Document` to build its own knowledge graph, not this project's pre-split 1000-char retrieval chunks. Wrap the full text as `langchain_core.documents.Document(page_content=full_text, metadata={"arxiv_id": paper.arxiv_id, "title": paper.title})`.
- **Embeddings**: reuse the `base_url` + `api_key="not-needed"` pattern from [`clients/embedding_client.py`](../../../clients/embedding_client.py) — see `references/custom_models.md` for how to wrap it as `LangchainEmbeddingsWrapper` for RAGAS.
- **LLM — currently missing**: RAGAS's testset generation (transforms, persona generation, query synthesis) and its default metrics both require a chat-completion LLM, and this project only has an embedding client so far (`clients/embedding_client.py` calls `/v1/embeddings`, not `/v1/chat/completions`). Before the `eval/` pipeline can run end-to-end, either the self-hosted server needs a chat-completions-capable model exposed, or a separate LLM client/env var (e.g. `LLM_BASE_URL`, `LLM_MODEL`) needs to be added, mirroring the embedding client's shape. Don't assume one exists — check `.env` and `clients/` first.
