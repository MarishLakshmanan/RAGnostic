# Evaluation

## Sample and dataset schema

`SingleTurnSample` represents one question/answer interaction to score:

```python
from ragas import SingleTurnSample, EvaluationDataset

sample = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany."],
    response="The capital of Germany is Berlin.",
    reference="Berlin",  # ground truth — required by reference-based metrics, optional otherwise
)

dataset = EvaluationDataset(samples=[sample])
```

`MultiTurnSample` is the equivalent for conversational/agent evaluations (list of turns instead of a single `user_input`/`response` pair) — used for the agent/tool-use metrics below. All samples in one `EvaluationDataset` must be the same type (all single-turn or all multi-turn).

You can also build a dataset from a Hugging Face dataset: `EvaluationDataset.from_hf_dataset(hf_ds)`.

If you generated a testset via `TestsetGenerator` (see `testset_generation.md`), run your RAG pipeline against each sample's `user_input` to fill in `response`/`retrieved_contexts`, keeping the synthesized `reference` — then wrap the results as `SingleTurnSample`s the same way.

## Running evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

`evaluate()` needs an LLM for any LLM-based metric — pass `llm=`/`embeddings=` explicitly, or metrics default to whatever's configured globally. Results are returned as a `Result` object (behaves like a dict of metric → score, and can be converted to a pandas DataFrame for per-sample scores).

## Metrics catalog

Most metrics are reference-free (judge the response/context against each other, not against ground truth) except where noted.

**Retrieval quality**
- `context_precision` — are the retrieved-and-ground-truth-relevant chunks ranked near the top? (needs `reference`)
- `context_recall` — does the retrieved context contain everything needed to support the reference answer? (needs `reference`)
- `context_entities_recall` — entity-level recall of retrieved context vs. reference (needs `reference`)
- `noise_sensitivity` — how much irrelevant retrieved content degrades the answer

**Generation quality**
- `faithfulness` — is every claim in `response` actually supported by `retrieved_contexts`? (reference-free — checks groundedness, not correctness)
- `answer_relevancy` / `response_relevancy` — does `response` actually address `user_input`? (reference-free)

**Multimodal** — `multimodal_faithfulness`, `multimodal_relevance` — same idea, for image+text contexts.

**Reference-based text comparison** (all need `reference`)
- `factual_correctness`, `semantic_similarity`, `bleu_score`, `chrf_score`, `rouge_score`, `exact_match`
- Reference-free string checks: `non_llm_string_similarity`, `string_present`

**Agent / tool use** — `topic_adherence`, `tool_call_accuracy`, `tool_call_f1`, `agent_goal_accuracy` — for `MultiTurnSample` conversations involving tool calls.

**NVIDIA metrics** — `answer_accuracy` (reference-based), `context_relevance`, `response_groundedness` — an alternate metric set with the same intent as the core ones above.

**General-purpose / custom criteria**
- `AspectCritic` — binary pass/fail judged against a described aspect (e.g. "is the answer harmful?")
- `SimpleCriteriaScoring`, rubric-based scoring, instance-specific rubrics — for scoring against a custom rubric without writing a full metric class.

**SQL** — `datacompy_score`, SQL query equivalence — for text-to-SQL pipelines specifically.

**Summarization** — dedicated metric for summary quality.

Full canonical list: `docs.ragas.io/en/stable/concepts/metrics/available_metrics/`.

## Lightweight custom metrics

For a one-off custom criterion, newer RAGAS versions offer `DiscreteMetric` + `llm_factory` instead of subclassing `Metric`:

```python
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

evaluator_llm = llm_factory("gpt-4o")  # or provider="openai"/"anthropic" with a custom client, see custom_models.md

metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="Evaluate if the summary is accurate and captures key information.\n\nResponse: {response}\n\nAnswer with only 'accurate' or 'inaccurate'.",
)

score = await metric.ascore(llm=evaluator_llm, response="...")
print(score.value, score.reason)
```

Use this for quick custom pass/fail or categorical judgments; use the classic metric classes above (`faithfulness`, `context_precision`, etc.) for standard RAG evaluation via `evaluate()`.
