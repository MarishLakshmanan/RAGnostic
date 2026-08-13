"""RAGAS quality test for single-hop RAG answers."""

from __future__ import annotations

import asyncio

from ragas import Dataset, experiment
from ragas.backends import InMemoryBackend, LocalCSVBackend
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from main import answer_question

MAX_CONCURRENT_ROWS = 4


def test_single_hop_quality(single_hop_rows, ragas_llm, ragas_embeddings, results_dir):
    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=ragas_llm, embeddings=ragas_embeddings
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ROWS)

    @experiment()
    async def run_single_hop(row: dict) -> dict:
        async with semaphore:
            result = answer_question(row["question"], use_graph=False)

            try:
                faithfulness_value = (
                    await faithfulness_metric.ascore(
                        user_input=row["question"],
                        response=result["answer"],
                        retrieved_contexts=result["contexts"],
                    )
                ).value
            except Exception as e:
                print(f"Error occurred while calculating faithfulness: {e}")
                faithfulness_value = float("nan")

            try:
                answer_relevancy_value = (
                    await answer_relevancy_metric.ascore(
                        user_input=row["question"], response=result["answer"]
                    )
                ).value
            except Exception as e:
                print(f"Error occurred while calculating answer relevancy: {e}")
                answer_relevancy_value = float("nan")

            try:
                reference_ids = set(row["arxiv_ids"])
                retrieved_ids = set(result["arxiv_ids"])
                recall_value = (
                    len(reference_ids & retrieved_ids) / len(reference_ids)
                    if reference_ids
                    else float("nan")
                )
            except Exception as e:
                print(f"Error occurred while calculating recall: {e}")
                recall_value = float("nan")

        return {
            "user_input": row["question"],
            "retrieved_contexts": result["contexts"],
            "response": result["answer"],
            "reference": row["answer"],
            "faithfulness": faithfulness_value,
            "answer_relevancy": answer_relevancy_value,
            "recall": recall_value,
            "group": "single_hop",
        }

    dataset = Dataset(
        name="single_hop", backend=InMemoryBackend(), data=single_hop_rows
    )
    experiment_result = asyncio.run(
        run_single_hop.arun(
            dataset,
            name="single_hop_results",
            backend=LocalCSVBackend(root_dir=str(results_dir)),
        )
    )

    df = experiment_result.to_pandas()

    assert df["faithfulness"].mean() >= 0.5
    assert df["answer_relevancy"].mean() >= 0.5
    assert df["recall"].mean() >= 0.5
