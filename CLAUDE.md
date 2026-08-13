# RAG-eval

Fetches arXiv papers for a corpus of LLM/agent-security papers, loads them into Postgres, chunks their PDFs for retrieval, and embeds the chunks into a vector store. Pipeline has six stages:

1. **Fetch** (`corpus/fetch_corpus.py`) → writes `corpus/papers.json`
2. **Load** (`store/db.py`) → reads a JSON file → upserts into Postgres `papers` table
3. **Chunk** (`chunking/parser.py`) → reads `papers` from Postgres → downloads each PDF → chunks it → filters out bibliography/reference-list chunks → upserts into Postgres `chunk` table
4. **Embed** (`embed/embed.py`) → reads unembedded `chunk` rows from Postgres → embeds them via an OpenAI-compatible endpoint → upserts vectors into Chroma → marks `chunk.embedded_at`
5. **Graph** (`graph/build_graph.py`) → reads embedded `chunk` rows → queries Chroma for near-neighbor chunks → upserts adjacency into Postgres `edges` table
6. **BM25 index** (`embed/build_bm25_index.py`) → reads `chunk` rows → builds a BM25 sparse index → saves it to `store/bm25_index/`

## `corpus/` package

### `corpus/fetch_corpus.py`

- `fetch_corpus(query, count, page_size=50, timeout=10.0)` — generator that queries the arXiv API (`http://export.arxiv.org/api/query`), parses the Atom feed, and yields `Paper` objects one at a time. Only yields papers whose `pdf_url` is verified reachable (HEAD request, GET fallback on 405); if some fraction of a page's PDFs are unreachable, it transparently fetches more pages until `count` reachable papers have been yielded or arXiv's results are exhausted. Streams via `yield` rather than collecting everything up front.
- `_fetch_page` retries transient network errors (timeouts, connection errors) up to 3x with exponential backoff — arXiv's public API is flaky under sustained sequential load.
- Sleeps `ARXIV_RATE_LIMIT_SECONDS` (3s) between page requests to respect arXiv's rate-limit guidance.
- `save_papers_to_json(papers, path)` — writes a list of `Paper` to disk via `PAPERS_ADAPTER.dump_json(...)` (bytes, UTF-8, handles `datetime` encoding).
- `__main__` block runs the full pipeline against a hardcoded LLM/agent-security query: fetch → `corpus/papers.json` → `save_papers_to_db(json_path)`.
- **Must be run as a module, not by file path**: `uv run python -m corpus.fetch_corpus` from the project root. Running `uv run python corpus/fetch_corpus.py` directly breaks the `from corpus.schemas import ...` / `from store.db import ...` absolute imports, since Python puts the script's own directory (not the project root) on `sys.path` when run by path.

### `corpus/schemas.py`

- `Paper` — the Pydantic model shared across fetching, JSON serialization, and DB loading (`arxiv_id`, `title`, `authors`, `summary`, `published: datetime`, `pdf_url`).
- `PAPERS_ADAPTER = TypeAdapter(list[Paper])` — single shared adapter for `list[Paper]` (de)serialization, used by both `fetch_corpus.py` (writing) and `store/db.py` (reading), so both sides agree on encoding (e.g. `datetime` ↔ ISO 8601) instead of each implementing it separately.

## `store/` package

Top-level package (sibling of `corpus/`, not nested under it) holding all persistence code.

- `models.py` — SQLAlchemy 2.0-style ORM, single shared `Base`/`metadata` for the whole schema:
  - `PaperRecord` → table `papers`. Surrogate `id` primary key; `arxiv_id` has a unique index (`ix_papers_arxiv_id`) used as the upsert conflict target. `authors` stored as Postgres `ARRAY(String)`, `published` as `DateTime(timezone=True)`.
  - `Chunk` → table `chunk`. Surrogate `id` primary key; `paper_id` is a `ForeignKey("papers.id", ondelete="CASCADE")`; unique on `(paper_id, chunk_index, strategy)` — the upsert conflict target, and what lets the same paper be re-chunked under a different strategy without colliding with existing rows. `embedded_at` (`DateTime(timezone=True)`, nullable) tracks whether/when a chunk has been embedded into Chroma — `NULL` means not yet embedded. `create_all` doesn't alter existing tables, so adding this column to an already-populated local `chunk` table requires a manual `ALTER TABLE chunk ADD COLUMN embedded_at TIMESTAMPTZ;`.
  - `Edge` → table `edges`. Surrogate `id` primary key; `source_chunk_id` is a `ForeignKey("chunk.id", ondelete="CASCADE")`, unique and indexed — one edge row per source chunk, so re-running graph-building for the same chunk updates rather than duplicates. `target_chunk_ids` stored as Postgres `ARRAY(Integer)` — the list of nearest-neighbor chunk ids for that source.
- `db.py` — engine/session setup and persistence:
  - Loads `DATABASE_URL` from `.env` via `python-dotenv` at import time; `engine`/`SessionLocal` created at module load.
  - `init_db()` — `Base.metadata.create_all(engine)`; additive/idempotent, safe to call even if some tables already exist.
  - `save_papers_to_db(json_path)` — reads and validates a JSON file via `PAPERS_ADAPTER` (from `corpus.schemas`), then does a single bulk `INSERT ... ON CONFLICT (arxiv_id) DO NOTHING ... RETURNING id`. Returns the count of newly inserted rows (re-running against an already-loaded file is a safe no-op — duplicates are skipped, not updated).
  - Imported as `from store.db import init_db, save_papers_to_db`.

## `chunking/` package

Turns each paper's PDF into rows in the `chunk` table. Two files, split so the chunking strategy is swappable independently of the DB-orchestration logic.

### `chunking/chunker.py`

- `ChunkFn = Callable[[bytes], list[str]]` — the contract every chunking strategy must satisfy: raw PDF bytes in, ordered list of chunk texts out. This is what makes strategies swappable — `chunking/parser.py` only depends on this signature, not on any particular implementation.
- `chunk_pdf_bytes(pdf_bytes, chunk_size=1000, chunk_overlap=200)` — the default/current strategy. Opens the PDF with PyMuPDF, extracts text page by page, joins all pages into one string (rather than splitting per page) so `RecursiveCharacterTextSplitter` (from `langchain_text_splitters`) can carry overlap naturally across what were page boundaries instead of truncating context at every page break. Strips literal NUL (`\x00`) bytes from the extracted text before splitting — some PDFs embed them in decoded text streams, and Postgres `TEXT` columns reject them outright.
- To add a new strategy: write another `bytes -> list[str]` function anywhere and pass it as `chunk_fn` to `iter_paper_chunks`/at the `chunking/parser.py` `__main__` call site — no changes needed elsewhere.

### `chunking/bibliography_filter.py`

Drops reference-list/bibliography chunks before they ever reach the `chunk` table, so every downstream consumer (dense embedding, BM25 indexing, graph building, eval generation) is automatically clean — no DB schema change needed, since a chunk that's never persisted is invisible to all of them. Motivated by `graph/build_graph.py`'s nearest-neighbor search over-representing bibliography↔bibliography edges (reference-list text across different papers is structurally near-identical and embeds unusually close together), which was polluting `eval/generate_testset.py`'s multi-hop sample generation.

- `locate_references_page(doc, page_texts)` — finds the page a paper's reference section starts on. First checks `doc.get_toc()` (PDF outline/bookmarks) for an entry titled "references"/"bibliography"/"related work"; if that's absent, scans `page_texts` from the **end backward** for a page containing a standalone heading line matching the same set of titles (scanning from the end avoids matching the word used mid-sentence in the body). Returns `None` if neither finds anything. Splits at page granularity, not exact character offset — a simplification, since this is a safety net rather than the sole filtering mechanism.
- `split_body_and_references(pdf_bytes)` — opens the PDF, extracts per-page text, calls `locate_references_page`, and returns `(body_text, references_text)` (`references_text` is `None` if no boundary was found). This re-parses the PDF independently of `chunker.py:extract_pdf_text` (which stays untouched, still used as-is by `eval/dataset.py`/`eval/generate_testset.py` for full-text extraction) rather than threading a new return value through that widely-used function's signature.
- `embed_in_batches(texts, batch_size=32)` — batches calls to `embed_texts` (`clients/embedding_client.py`), since the self-hosted embedding server rejects batches larger than 32 (same constant `embed/embed.py` documents).
- `filter_bibliography_chunks(pdf_bytes, chunks, threshold=SIMILARITY_THRESHOLD)` — the main entry point. If no reference section can be located, returns `chunks` unchanged. Otherwise splits `references_text` into small fragments via `recursive_chunk_text` at `chunk_size=300` (deliberately finer-grained than the main chunking strategy, since these fragments only exist to approximate individual citations for comparison, not to be retrieved), embeds both the fragments and the candidate `chunks`, and computes cosine similarity via `numpy`: normalize both embedding matrices, one matrix multiply → `(num_chunks, num_reference_fragments)` similarity matrix → `.max(axis=1)` gives each chunk's best-match score. Drops any chunk whose max score exceeds `threshold` — max (not average) aggregation, since a bibliography-like chunk should be flagged for closely resembling *any single* citation, even if it's dissimilar to the rest of the reference list.
- Module-level: reads `SIMILARITY_THRESHOLD` from `config.yaml`'s `bibliography_filter.similarity_threshold` key at import time (same pattern as `chunking/chunker.py`'s module-level config load). No `enabled` flag — filtering always runs; a paper only skips it when no reference section can be located.

### `chunking/parser.py`

- `iter_paper_chunks(chunk_fn=chunk_pdf_bytes, timeout=10.0)` — generator that `SELECT`s every row from `papers`, checks each `pdf_url` for reachability (reuses `_is_pdf_reachable` from `corpus.fetch_corpus` rather than duplicating it), downloads reachable PDFs, runs them through `chunk_fn`, filters the result through `bibliography_filter.filter_bibliography_chunks`, and yields `(paper, chunks)` pairs. Read-only against the DB — safe to run without writing anything, though it now also depends on the embedding server being reachable (for the bibliography filter's similarity check).
- `save_chunks_to_db(paper_id, chunks, strategy)` — bulk `INSERT ... ON CONFLICT (paper_id, chunk_index, strategy) DO NOTHING ... RETURNING id`, mirroring `save_papers_to_db`'s pattern. `strategy` is recorded per row (defaults to `chunk_fn.__name__` at the call site) so multiple strategies can be run against the same paper and compared later instead of overwriting each other.
- `__main__` block: `init_db()` → `iter_paper_chunks()` → `save_chunks_to_db(...)` per paper, printing progress. Run via `uv run python -m chunking.parser` from the project root (same implicit-namespace-package/absolute-import constraint as `corpus/fetch_corpus.py`).

## `clients/` package

Top-level package (sibling of `corpus/`, `store/`, `chunking/`) holding thin HTTP clients for external services. Each file loads its own env vars via `python-dotenv` independently — no shared config module.

- `embedding_client.py` — module-level `openai.OpenAI` client pointed at `EMBEDDING_BASE_URL` (an OpenAI-compatible `/v1/embeddings` server, no API key — a placeholder string is passed since the SDK requires a non-empty one). `embed_texts(texts)` calls `client.embeddings.create(input=texts, model=EMBEDDING_MODEL)` and returns the embedding vectors in the same order as the input.
- `chroma_client.py` — module-level `chromadb.HttpClient` pointed at `CHROMA_HOST`/`CHROMA_PORT`. `get_collection()` returns `client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={...})` — idempotent, safe to call repeatedly. Collection metadata records `embedding_model` and `topic` so the collection self-documents what it holds (matters if the embedding model ever changes, since different models produce incompatible vector spaces and would need a new collection). Note: Chroma only applies `metadata` on first creation — it's ignored on subsequent `get_or_create_collection` calls against an already-existing collection.
- `llm_client.py` — module-level `langchain_openai.ChatOpenAI` client (`chat_llm`), config driven by `LLM_MODEL`/`LLM_BASE_URL` env vars. The client is OpenAI-API-shaped, meant to be pointed at any OpenAI-compatible chat endpoint (e.g. LM Studio running locally, or OpenRouter) rather than a specific provider. `API_KEY` resolves from `LLM_API_KEY` if set, else falls back to a placeholder string (`"LM_STUDIO"`) for local servers that don't check the key. Also exports a raw `openai.OpenAI` client (`client = OpenAI(base_url=BASE_URL, api_key=API_KEY)`) alongside `chat_llm`, added because RAGAS's `llm_factory` requires a raw SDK client rather than a LangChain wrapper — this is what `tests/conftest.py`'s `ragas_llm` fixture consumes.
- `bm25_client.py` — lazy-loaded singleton for the on-disk BM25 index, mirroring `chroma_client.py`'s shape but deferring the actual load: `get_bm25_retriever()` calls `bm25s.BM25.load(BM25_INDEX_DIR, load_corpus=True)` on first use and caches the result in a module-level global, rather than loading at import time. Deferred because the index (`store/bm25_index/`) may not exist yet when this module is imported, and is only actually needed when `retrieval_strategy` is `"hybrid"`.

## `embed/` package

Embeds each unembedded `chunk` row and stores the vector in Chroma. Single file (`embed.py`), mirroring `chunking/parser.py`'s orchestration style.

- `iter_unembedded_chunk_batches(batch_size=32)` — generator that repeatedly runs `SELECT * FROM chunk WHERE embedded_at IS NULL LIMIT batch_size` and yields the batch. Because the caller marks each batch embedded before the next iteration runs, re-issuing the identical query naturally advances through the table — no `OFFSET`/keyset cursor needed. `batch_size` defaults to 32 to match the local embedding server's maximum batch size.
- `mark_chunks_embedded(chunk_ids)` — bulk `UPDATE chunk SET embedded_at = now() WHERE id IN (...)`. Unlike the `on_conflict_do_nothing` insert pattern used elsewhere, this updates existing rows, so it's a plain `UPDATE`.
- `embed_and_upsert_batch(chunks)` — embeds the batch via `embed_texts`, upserts into Chroma with `ids=[str(chunk.id) for chunk in chunks]` (no `documents`/`metadatas` — retrieval looks up chunk text back in Postgres by id) via `collection.upsert(...)`, then calls `mark_chunks_embedded`. Order is load-bearing: embed → Chroma upsert → mark embedded, so a crash between the last two steps just leaves `embedded_at` NULL and the batch is safely retried (Chroma's `upsert` is idempotent by id — redoing it is a harmless no-op, not corruption).
- `__main__` block: `init_db()` → loop over `iter_unembedded_chunk_batches()` → `embed_and_upsert_batch(...)` per batch, printing progress. Run via `uv run python -m embed.embed` from the project root (same implicit-namespace-package/absolute-import constraint as the other pipeline entrypoints).

### `embed/build_bm25_index.py`

Builds a BM25 sparse index over chunk text and persists it to disk, for the `hybrid` retrieval strategy in `main.py`. Lives in `embed/` (a build/indexing step, like `embed.py`) rather than `store/` — the saved index artifact itself lives in `store/bm25_index/`, since that's data/state, like the Postgres DB and Chroma collection.

- `build_bm25_index(strategy="chunk_pdf_bytes", index_dir="store/bm25_index")` — `SELECT`s `Chunk` rows for a given chunking strategy, builds a corpus of `{"id": chunk.id, "text": chunk.content}` dicts, tokenizes via `bm25s.tokenize(..., stopwords="en")`, builds and fits a `bm25s.BM25()`, then `.save(index_dir, corpus=corpus)`. Saving the corpus as `{"id", "text"}` dicts (rather than just the index) is what lets a later `.retrieve()` call recover the original chunk id directly from the result, without a separate lookup table. Raises `ValueError` if no chunks exist for the given strategy.
- `__main__` block: `init_db()` → `build_bm25_index()`, printing the indexed count. Run via `uv run python -m embed.build_bm25_index` from the project root (same implicit-namespace-package/absolute-import constraint as the other pipeline entrypoints).

## `graph/` package

Builds a chunk-adjacency graph by querying Chroma for near-neighbor chunks and persisting the result as edges in Postgres. Single file (`build_graph.py`), mirroring `embed/embed.py`'s orchestration style.

- `iter_embedded_chunk_batches(batch_size=32)` — generator yielding batches of embedded chunks (`Chunk.embedded_at IS NOT NULL`) that don't yet have a row in `edges` (`Chunk.id NOT IN (SELECT source_chunk_id FROM edges)`) — same incremental-progress pattern as `embed.embed`'s batch generator.
- `build_and_save_edges_batch(chunks, top_n=5, min_index_gap=3)` — embeds the batch's text via `embed_texts` (reused from `clients/embedding_client.py`), then for each chunk queries Chroma for its `top_n` nearest neighbors, excluding same-paper chunks within `min_index_gap` of the source chunk's index (so trivially-adjacent chunks from the same paper don't count as "related" — the interesting edges are either cross-paper or non-adjacent same-paper). Bulk `INSERT ... ON CONFLICT (source_chunk_id) DO UPDATE` into `edges`, so re-running is safe and refreshes stale neighbor lists rather than duplicating rows.
- `__main__` block: `init_db()` → loop over `iter_embedded_chunk_batches()` → `build_and_save_edges_batch(...)` per batch, printing progress. Run via `uv run python -m graph.build_graph` from the project root (same implicit-namespace-package/absolute-import constraint as the other pipeline entrypoints).

## `eval/` package

Generates a synthetic single-hop/multi-hop QA testset by calling an LLM directly — no RAGAS `TestsetGenerator`/`KnowledgeGraph` involved (an earlier RAGAS-based approach was abandoned).

### `eval/dataset.py`

- `iter_paper_documents(timeout: float = 10.0)` — generator that walks `papers` in Postgres, checks each PDF for reachability (reuses `_is_pdf_reachable` from `corpus/fetch_corpus`), downloads reachable PDFs, extracts full text via `extract_pdf_text` (from `chunking/chunker`), and yields `Document` objects one at a time (reusable pattern, like `iter_paper_chunks`). Metadata includes `arxiv_id` and `title` for later reference.

### `eval/generate_testset.py`

Two independent sample groups, each written to its own JSONL file (not combined). Uses `clients/llm_client.py`'s `chat_llm` (`ChatOpenAI` against an OpenAI-compatible endpoint) via `.with_structured_output(...)` against Pydantic response schemas (`SingleHopResponse`, `MultiHopResult`, defined in this file) rather than free-form prompting, so responses parse reliably. System prompts live in `eval/prompts.py` (`SINGLE_HOP_SYSTEM_PROMPT`, `MULTI_HOP_SYSTEM_PROMPT`).

- **Single-hop** (`generate_single_hop_samples(count)`): repeatedly picks a random paper not yet used (`get_random_unused_paper`, `SELECT ... ORDER BY random()` excluding already-used ids), fetches its full text (`fetch_paper_text`, same reachability-check-then-`extract_pdf_text` pattern as `eval/dataset.py`), and asks the LLM for exactly 2 question/answer/ground-truth-excerpt triples per paper. Each paper is used at most once, so no two single-hop samples share a source paper. Stops at `count` samples or when papers are exhausted.
- **Multi-hop** (`generate_multi_hop_samples(count, max_attempts=count*10)`): repeatedly picks a random row from `edges` (`get_random_cross_paper_edge`), loads the source chunk and its target chunks, and filters targets down to ones whose `paper_id` differs from the source's — `edges` rows aren't guaranteed cross-paper (see `graph/build_graph.py`), so rows with fewer than 2 cross-paper targets are discarded and re-sampled. The 2-3 remaining excerpts go to the LLM with a prompt asking for a question answerable only by combining them; if the LLM reports one excerpt alone would suffice (`answerable_from_single_excerpt`), the sample is discarded rather than kept. Bounded by `max_attempts` so a sparse `edges` table can't spin forever.
- `save_samples_to_jsonl(samples, path)` — one JSON object per line; each sample carries `group` ("single_hop"/"multi_hop"), `question`, `answer`, `reference_contexts` (list of source excerpts), and `arxiv_ids` (provenance).
- `run_single_hop_generation(count)`/`run_multi_hop_generation(count)` — each calls its respective generator and writes the result to its own file: `eval/single_hop_samples.jsonl` / `eval/multi_hop_samples.jsonl`. No `argparse` — these are plain functions, not CLI flags.
- `__main__` block: `init_db()` then one hardcoded call (currently `run_multi_hop_generation(count=20)`, with the single-hop call commented out) — a manual/ad-hoc entrypoint you edit directly, not flag-driven. Run via `uv run python -m eval.generate_testset` from the project root (same implicit-namespace-package/absolute-import constraint as the other pipeline entrypoints).
- **Note**: `eval/testset.jsonl` — the file `tests/conftest.py` actually reads — is not produced by this script. It must be assembled from `single_hop_samples.jsonl`/`multi_hop_samples.jsonl` some other way (manually, or a step not currently in the repo).

## `main.py`

The RAG pipeline entrypoint — takes a question, retrieves context, and generates an answer. Not part of the corpus-building pipeline stages above; this is the query side. Every retrieval/fusion function is wrapped in `langfuse`'s `@observe()` decorator for tracing.

- Module-level: reads `RETRIEVAL_STRATEGY` from `config.yaml`'s `retrieval_strategy.name` at import time (same module-level config-load pattern as `chunking/chunker.py`).
- `retrieve_top_chunks(query, top_k=5)` — embeds `query` via `embed_texts` (from `clients/embedding_client.py`), queries the Chroma collection (`clients/chroma_client.py:get_collection`) for the `top_k` nearest chunk ids, then looks up their text back in Postgres (Chroma stores no `documents`/`metadatas`, only ids — same as noted in `embed/embed.py`), reordered to match Chroma's similarity ranking. Returns `list[ChunkModel]` (from `corpus/schemas.py`), not raw `Chunk` ORM objects — `langfuse`'s `@observe()` can't serialize SQLAlchemy objects, so results are converted via `ChunkModel.model_validate(...)` before returning.
- `retrieve_bm25_chunks(query, top_k=5)` — the BM25 sparse-search counterpart to `retrieve_top_chunks`. Tokenizes the query via `bm25s.tokenize`, retrieves the top-k matches from `clients/bm25_client.py:get_bm25_retriever()` (each a `{"id", "text"}` dict, since that's how `embed/build_bm25_index.py` saved the corpus), pulls the chunk ids out in rank order, then looks the chunks up fresh in Postgres by id — same "fetch by id from Chroma/BM25, reorder to match the store's rank, look up authoritative content in Postgres" shape as `retrieve_top_chunks`. Also returns `list[ChunkModel]`.
- `reciprocal_rank_fusion(ranked_lists, top_k=5, k=60)` — merges multiple ranked `list[ChunkModel]`s (e.g. dense + BM25 results) into one via RRF: each chunk's score is `sum(1 / (k + rank + 1))` across every list it appears in, summed across lists, then sorted descending and deduplicated. Used to combine `retrieve_top_chunks` and `retrieve_bm25_chunks` for the `hybrid` retrieval strategy.
- `expand_with_graph(chunk)` — given a single chunk, looks up its row in `edges` and returns `[chunk]` plus every one of its stored `target_chunk_ids` neighbors (no cross-paper filtering, unlike `eval/generate_testset.py`'s multi-hop sampling — this just uses whatever `graph/build_graph.py` already stored). Returns `[chunk]` alone if it has no edge row.
- `build_context(chunks)` — joins chunk texts with a `"\n\n---\n\n"` delimiter.
- `get_arxiv_ids_for_chunks(chunks)` — resolves a list of retrieved chunks (`ChunkModel`, or the `Chunk`/`ChunkModel` mix `expand_with_graph` can produce — both expose `.paper_id`) to their source papers' arxiv ids via one `PaperRecord` lookup keyed on the chunks' unique `paper_id`s. Exists so retrieval can be scored against ground-truth papers (see `tests/` below) without threading a DB session through the test files themselves.
- `answer_question(question, use_graph=False, top_k=5)` — the pipeline entrypoint, dispatching on retrieval strategy:
  - `use_graph=True` (explicit param) always forces graph expansion, regardless of `config.yaml` — kept as a hard override because `tests/test_single_hop.py`/`tests/test_multi_hop.py` pass it explicitly to force a specific pipeline for each test file.
  - Otherwise branches on `RETRIEVAL_STRATEGY`: `"graph"` → same graph-expansion behavior (`retrieve_top_chunks` then `expand_with_graph` on the best match) so config alone can select it without also passing `use_graph`; `"hybrid"` → `retrieve_top_chunks` + `retrieve_bm25_chunks`, merged via `reciprocal_rank_fusion`; `"dense"`/`"none"`/anything else → plain `retrieve_top_chunks` (today's default).
  - Builds context from whichever chunk set results, formats `SYSTEM_PROMPT_TEMPLATE` with it, and calls `chat_llm.invoke(...)` (from `clients/llm_client.py`) with that system message plus the question as the user message. Returns `{"answer": str, "contexts": list[str], "arxiv_ids": list[str]}` — `contexts` are included so callers (e.g. RAGAS-based evaluation) can score the answer against exactly what was retrieved, without re-deriving it; `arxiv_ids` (via `get_arxiv_ids_for_chunks`) are the source papers of the retrieved chunks, used by `tests/`'s recall metric.
- `__main__` block: runs `answer_question` against a sample question and prints the answer. Run via `uv run python main.py` or `uv run python -m main` — no absolute-import indirection needed since it's a top-level module, not a package member.

## `tests/` package

RAGAS-based quality gates for `main.py:answer_question`, run with pytest. `pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["."]` so `main`/`clients`/etc. import correctly regardless of how pytest is invoked (unlike the other pipeline entrypoints, no `-m` invocation is needed).

- `conftest.py` — `_load_testset_group(group)` reads `eval/testset.jsonl`, filters by `group`, and truncates to the first 5 rows, exposed via the `single_hop_rows`/`multi_hop_rows` fixtures. `ragas_llm` (session-scoped) is `llm_factory(LLM_MODEL, provider="openai", client=llm_openai_client)` — RAGAS's modern `InstructorBaseRagasLLM` interface, built from `clients/llm_client.py`'s raw `client` (not the LangChain-wrapped `chat_llm`, which `llm_factory` can't consume directly). `ragas_embeddings` (session-scoped) is `embedding_factory(provider="openai", model=EMBEDDING_MODEL, client=embedding_openai_client)`, reusing `clients/embedding_client.py`'s existing raw `openai.OpenAI` client directly — no LangChain wrapper needed on the embedding side. `ragas_run_config` (`RunConfig(max_workers=4, timeout=1800, log_tenacity=True)`, tuned for OpenRouter) still exists but is currently **unused** — neither test file's `@experiment`-based code path takes a `RunConfig`. `results_dir` creates/returns `tests/results/` (gitignored — run artifacts, not source).
- `test_single_hop.py` / `test_multi_hop.py` — same shape in both. Build `Faithfulness(llm=ragas_llm)` and `AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)` (`ragas.metrics.collections` — modern per-instance metric classes, replacing the deprecated `ragas.metrics.faithfulness`/`answer_relevancy` singletons and `ragas.evaluate()`). An `@experiment()`-decorated async function (`run_single_hop`/`run_multi_hop`) calls `answer_question(row["question"], use_graph=False|True)`, then scores the result with both RAGAS metrics via `.ascore(...)` plus a third, non-LLM **recall** metric — each wrapped in its own `try/except → float("nan")` so one metric failing doesn't blank out the others for that row (preserves the old `evaluate(raise_exceptions=False)` per-metric resilience; `@experiment`'s own exception handling is per-*row*, not per-metric, and would otherwise drop the whole row on any single failure). Recall is computed as `|set(row["arxiv_ids"]) ∩ set(result["arxiv_ids"])| / |set(row["arxiv_ids"])|` — the fraction of the testset row's ground-truth source paper(s) that were actually among the retrieved chunks' papers. Deliberately not RAGAS's `context_recall` metric, which would need another LLM call to judge context against the reference answer; since the testset already records which paper(s) a question's answer should come from (`arxiv_ids`, from `eval/generate_testset.py`), a paper-level set-intersection is a free, deterministic proxy. Both wrap their per-row work in `asyncio.Semaphore(MAX_CONCURRENT_ROWS=4)` since `.arun()` has no built-in concurrency cap — though `answer_question()` itself is a synchronous blocking call, so it doesn't actually run concurrently across rows even with the semaphore; only the two `.ascore()` awaits genuinely overlap. Input rows are wrapped in a `Dataset(name=..., backend=InMemoryBackend(), data=...)`; output is written by `ExperimentWrapper.arun(dataset, name=..., backend=LocalCSVBackend(root_dir=str(results_dir)))`, which auto-saves to **`tests/results/experiments/{name}.csv`** (`LocalCSVBackend`'s fixed subdirectory, not `tests/results/{name}.csv` directly). `.to_pandas()` on the returned `Experiment` gives the same `user_input`/`retrieved_contexts`/`response`/`reference`/`faithfulness`/`answer_relevancy`/`recall` shape as before; both tests assert all three metrics' means are `>= 0.5`.
- Run via `uv run pytest tests/ -v` from the project root. Each test makes ~5 live RAG pipeline calls plus RAGAS judge LLM/embedding calls against OpenRouter — expect real latency and API usage, not an instant/free run.

## `web/` package

Renders `tests/results/*/` (RAGAS experiment output) into a static HTML report under `web/dist/` — a local dev tool for browsing eval runs, not part of the corpus/eval pipeline.

### `web/build.py`

- `SCORE_COLUMNS = ("faithfulness", "answer_relevancy", "recall")` — single source of truth for which experiment-CSV columns get tiered score-badge rendering and a mean-score summary; `_build_table` and `results.html` both key off this tuple, so adding a new metric column to the CSV only requires adding its name here (plus the two matching `results.html` spots noted below).
- `_score_tier(score)` — buckets a 0-1 score into `good`/`mid`/`poor`/`unknown` for badge coloring (`>=0.8` / `>=0.5` / below / `None`/NaN).
- `_clean_value(value)` — replaces NaN/NaT with `None` so row dicts survive `tojson` round-tripping into the page's embedded JSON (raw NaN isn't valid JSON).
- `_parse_contexts(raw)` — `ast.literal_eval`s the `retrieved_contexts` CSV cell (a Python-list-repr string like `"['a', 'b']"`) into an actual `list[str]`, falling back to a single-item list on parse failure so malformed data doesn't crash the build.
- `_format_timestamp(raw)` — treats `config.yaml`'s `timestamp` as UTC (matches how `tests/conftest.py`'s `results_dir` fixture writes it via `datetime.now(timezone.utc)`), converts to local time, and formats as `"%d, %b %y"` (e.g. `"11, Aug 26"`).
- `_build_table(csv_path)` — loads one experiment CSV via `pandas.read_csv`, builds per-row dicts (parsed `retrieved_contexts`, per-score CSS tiers), and computes per-table mean `faithfulness`/`answer_relevancy`/`recall` (`mean_scores`, labeled `"Avg ..."`, driven by `SCORE_COLUMNS`) plus row count.
- `discover_results(results_dir)` — iterates `tests/results/*/`, reads each `config.yaml`, and splits it into the root-level `name`/`changes`/`timestamp` keys versus the nested strategy dicts (`chunking_strategy`, `retrieval_strategy`, `reranking_strategy`). Display name falls back to the folder name if `config.yaml` has no `name`. Attaches `_build_table(...)` results for every CSV in that result's `experiments/` folder.
- `render_site(results, output_dir)` — renders `results.html`/`about.html` via a `jinja2.Environment(FileSystemLoader(...))` and writes them plus an `index.html` (a copy of the results page, so it doubles as the landing page) to `web/dist/`.
- `__main__` block: `discover_results(RESULTS_DIR)` → `render_site(...)`, printing progress. Run via `uv run python -m web.build` from the project root (same implicit-namespace-package/absolute-import constraint as the other pipeline entrypoints).

### `web/templates/` (Jinja2, template inheritance)

- `base.html` — shared shell: nav bar, the full CSS design system (color tokens, card/badge/table styles), and styling for the fixed right-side row-detail panel.
- `about.html` — extends `base.html`, empty content block (placeholder page).
- `results.html` — extends `base.html`. Each result renders as a collapsible `<details>` (strategy pills + local timestamp in the summary; inside, a config-card grid including a highlighted "Changes" card sourced from `config.yaml`'s `changes` key). Each CSV renders as a nested collapsible `<details>` with mean-score badges (marked with an "x̄" mean cue plus a tooltip) in its own summary. Table rows are fully clickable (`tr.row-clickable`, each carrying its data as JSON in a `data-row` attribute via `{{ row | tojson }}`) and open a fixed side panel showing the full, un-truncated `user_input`/`retrieved_contexts`/`response`/`reference` for that row — implemented via a small inline `<script>` at the bottom of the file (vanilla JS, no build step or framework). Two spots must stay in sync with `web/build.py`'s `SCORE_COLUMNS` when a metric column is added/removed: the `{% elif col in (...) %}` check that renders a table cell as a tiered score badge instead of plain text, and the JS `scoreLabels` map that titles each score badge in the side panel (currently `{faithfulness, answer_relevancy, recall}` in both places).

### Output

`web/dist/` holds the generated `index.html`/`about.html`/`results.html` — already covered by the repo's existing `dist/` gitignore rule, so it's never committed.

### Dependencies

`jinja2` is a direct entry in `pyproject.toml`'s `dependencies` (previously only pulled in transitively via `langchain-core`). `pandas` (dev dependency) and `pyyaml` (main dependency) were already present and are reused as-is.

## `config.yaml`

Read independently, at module load time, by several modules — each opens the file itself (no shared config-loading module):

- `chunking_strategy.name` — read by `chunking/chunker.py` to dispatch `chunk_pdf_bytes`'s chunking strategy (currently `"recursive_chunker"` is the only real case; the `match` statement's default case points to the same function).
- `retrieval_strategy.name` — read by `main.py` as `RETRIEVAL_STRATEGY`. Valid values: `"graph"`, `"dense"`, `"hybrid"`, `"none"` (see `main.py`'s `answer_question` above for the dispatch).
- `reranking_strategy.name` — declared (`"cross_encoder"` or `"none"`) but not yet read or implemented anywhere; also used by `tests/conftest.py`/`web/build.py` purely for naming/displaying result runs, not for driving behavior.
- `bibliography_filter.similarity_threshold` — read by `chunking/bibliography_filter.py`. No `enabled` flag; filtering always runs, and only skips a paper when no reference section could be located. Tunable — expected to need empirical adjustment by spot-checking dropped vs. kept chunks.

## Local dev DB

Postgres runs in Docker on `localhost:5432` (`devuser`/`devpassword`/`devdb`). Connection string lives in `.env` as `DATABASE_URL=postgresql+psycopg://...` (uses the `psycopg` v3 driver, not `psycopg2`).

## Local dev vector store

Chroma runs in Docker on `localhost` (host/port configurable). `.env` holds `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`, plus `EMBEDDING_BASE_URL`/`EMBEDDING_MODEL` for the self-hosted OpenAI-compatible embedding server. None of these require an API key.

## Local BM25 index

`store/bm25_index/` is an on-disk directory artifact, not a service (unlike Postgres/Chroma) — built by `embed/build_bm25_index.py`, loaded by `clients/bm25_client.py`, gitignored (build artifact, like `tests/results/`). No automatic re-indexing trigger: it must be rebuilt manually (`uv run python -m embed.build_bm25_index`) after `chunking.parser` adds new chunks, the same way `embed.embed`/`graph.build_graph` need re-running incrementally.

## Don'ts

1. Don't track plan.md, it's my scratchpad
