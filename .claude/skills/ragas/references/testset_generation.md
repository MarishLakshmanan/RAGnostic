# Testset generation

RAGAS builds a synthetic testset by constructing a `KnowledgeGraph` from your documents, enriching it, then synthesizing question/context/reference triples from it.

## Pipeline stages

1. **Knowledge graph construction** — each input document becomes a `Node` (`NodeType.DOCUMENT`), which gets chunked into hierarchical child nodes (`NodeType.CHUNK`) by a splitter.
2. **Transforms (enrichment)** — a pipeline of extractors/relationship-builders runs over the graph:
   - Extractors pull info into node properties: `NER` (named entities), keyphrases, summaries, embeddings. LLM-based extractors subclass `LLMBasedExtractor`; rule-based ones subclass `Extractor`.
   - Relationship builders connect nodes based on extracted properties — e.g. `JaccardSimilarityBuilder` links nodes whose entity sets overlap. This is what makes multi-hop questions possible: the synthesizer picks *related* node pairs, not random ones.
   - `default_transforms(documents=docs, llm=llm, embedding_model=embeddings)` returns a sensible default pipeline; `apply_transforms(kg, transforms)` runs it.
3. **Personas** — `Persona` objects describe types of users who'd query this content (e.g. "a security researcher", "a first-time reader"). Generated automatically (`num_personas` on `generate(...)`, default 3) or supplied via `persona_list` on `TestsetGenerator`.
4. **Scenario generation** — for each query synthesizer, `_generate_scenarios()` picks target node(s) from the graph plus a query length (short/medium/long), a query style (web-search-like vs conversational), and a persona.
5. **Sample synthesis** — `_generate_sample()` turns each scenario into a `TestsetSample` (question, reference contexts, reference answer) using the LLM.

## Simple path

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
```

`docs` is a sequence of `langchain_core.documents.Document`. Internally this builds the knowledge graph, applies default transforms, and uses the default query distribution — no manual KG/transform/distribution setup required. There's also `generate_with_llamaindex_docs` for LlamaIndex documents, and `generate_with_chunks` if you already have pre-chunked content and want to skip RAGAS's internal chunking (pass chunks in directly, treated as `NodeType.CHUNK`).

## Full control path — needed to cover all four query types

The default `query_distribution` only covers three types (50% SingleHopSpecific / 25% MultiHopAbstract / 25% MultiHopSpecific) — it omits `SingleHopAbstractQuerySynthesizer`. To include all four, build the distribution explicitly:

```python
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    SingleHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset import TestsetGenerator

kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata},
        )
    )

transforms = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
apply_transforms(kg, transforms)

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
    (SingleHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
]

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg)
testset = generator.generate(testset_size=40, query_distribution=query_distribution)
```

Each entry in `query_distribution` is a `(QuerySynthesizer, weight)` pair; weights should sum to 1.0. `generate(...)` samples synthesizers according to these weights until `testset_size` samples are produced.

## `TestsetGenerator` reference

```python
TestsetGenerator(
    llm: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings,
    knowledge_graph: KnowledgeGraph = KnowledgeGraph(),   # empty by default — build/populate before generate()
    persona_list: Optional[List[Persona]] = None,          # auto-generated if omitted
    llm_context: Optional[str] = None,                     # extra context appended to generation prompts
)
```

Factory methods `TestsetGenerator.from_langchain(llm, embedding_model, ...)` and `.from_llama_index(llm, embedding_model, ...)` build the wrapper objects for you from raw LangChain/LlamaIndex LLM+embeddings instances — equivalent to wrapping manually with `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` (see `custom_models.md`).

Key `generate()` parameters: `testset_size` (sample count), `num_personas` (default 3, ignored if `persona_list` was set on the generator), `query_distribution`, `with_debugging_logs`, `return_executor` (returns an `Executor` with `.cancel()`/`.results()` instead of blocking).

Raises `ValueError` if no LLM or embedding model was provided either at construction or call time.

## Output

`generate(...)` / `generate_with_langchain_docs(...)` return a `Testset` — a collection of `TestsetSample` objects. Each sample carries the synthesized `user_input` (question), the `reference_contexts` it was grounded in, and a `reference` (expected answer) — the same shape `EvaluationDataset`/`SingleTurnSample` expects for evaluation, so a generated `Testset` converts directly into an eval input once you've run your own RAG pipeline against each `user_input` to fill in `response`/`retrieved_contexts`.
