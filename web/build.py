"""Render tests/results/ into a static HTML site under web/dist/ using Jinja2."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

RESULTS_DIR = Path("tests/results")
TEMPLATES_DIR = Path(__file__).parent / "templates"
DIST_DIR = Path(__file__).parent / "dist"

SCORE_COLUMNS = ("faithfulness", "answer_relevancy", "recall")


def _score_tier(score: float | None) -> str:
    """Bucket a 0-1 score into a CSS tier name for badge coloring.

    Args:
        score (float | None): The score to bucket, or None if unavailable.

    Returns:
        str: "good" (>=0.8), "mid" (>=0.5), "poor" (<0.5), or "unknown" if score is None/NaN.
    """
    if score is None or pd.isna(score):
        return "unknown"
    if score >= 0.8:
        return "good"
    if score >= 0.5:
        return "mid"
    return "poor"


def _clean_value(value: Any) -> Any:
    """Replace NaN/NaT with None so the value round-trips through JSON safely.

    Args:
        value (Any): A raw pandas cell value.

    Returns:
        Any: The value unchanged, or None if it was NaN/NaT.
    """
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _parse_contexts(raw: Any) -> list[str]:
    """Parse a retrieved_contexts cell into a list of excerpt strings.

    Args:
        raw (Any): The raw cell value, typically a Python-list-repr string like "['a', 'b']".

    Returns:
        list[str]: The parsed excerpts, or a single-item list with the raw value stringified
            if it can't be parsed as a list.
    """
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    return [str(raw)]


def _build_table(csv_path: Path) -> dict[str, Any]:
    """Load one experiment CSV into a template-ready table structure.

    Args:
        csv_path (Path): Path to the CSV file to load.

    Returns:
        dict[str, Any]: `filename`, `row_count`, `columns`, `rows` (list of column->value dicts,
            with `retrieved_contexts` parsed into a list of excerpts), and per-score-column mean
            value + CSS tier (`mean_scores`: list of {name, mean, tier}).
    """
    frame = pd.read_csv(csv_path)
    columns = [col for col in frame.columns if col != "group"]

    rows = []
    for _, row in frame.iterrows():
        record = {
            col: _clean_value(row[col]) for col in columns if col != "retrieved_contexts"
        }
        record["retrieved_contexts"] = _parse_contexts(row["retrieved_contexts"])
        record["score_tiers"] = {col: _score_tier(row.get(col)) for col in SCORE_COLUMNS}
        rows.append(record)

    mean_scores = []
    for col in SCORE_COLUMNS:
        mean = frame[col].mean(skipna=True) if col in frame.columns else None
        mean_scores.append(
            {
                "name": "Avg " + col.replace("_", " ").title(),
                "mean": None if mean is None or pd.isna(mean) else round(float(mean), 3),
                "tier": _score_tier(mean),
            }
        )

    return {
        "filename": csv_path.name,
        "row_count": len(frame),
        "columns": columns,
        "rows": rows,
        "mean_scores": mean_scores,
    }


def _format_timestamp(raw: Any) -> str | None:
    """Format a config timestamp (assumed UTC) as a short local date like "11, Aug 26".

    Args:
        raw (Any): The raw `timestamp` value from config.yaml, expected to be a
            datetime as parsed by PyYAML (naive, but written in UTC by the test suite).

    Returns:
        str | None: The formatted local date, or None if `raw` isn't a datetime.
    """
    if not isinstance(raw, datetime):
        return None
    if raw.tzinfo is None:
        raw = raw.replace(tzinfo=timezone.utc)
    return raw.astimezone().strftime("%d, %b %y")


def discover_results(results_dir: Path) -> list[dict[str, Any]]:
    """Read every result folder's config and experiment CSVs into a template-ready structure.

    Args:
        results_dir (Path): Directory containing one subfolder per result run.

    Returns:
        list[dict[str, Any]]: One entry per result, each with `name` (config `name` if set,
            else the folder name), `config` (raw nested dict of strategy sections), `changes`
            (the config's `changes` note, if any), `timestamp` (formatted local date, if any),
            and `tables` (per-CSV structures from _build_table).
    """
    results = []
    for result_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        config_path = result_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        csv_paths = sorted((result_dir / "experiments").glob("*.csv"))
        tables = [_build_table(csv_path) for csv_path in csv_paths]

        strategies = {k: v for k, v in config.items() if isinstance(v, dict)}

        results.append(
            {
                "name": config.get("name") or result_dir.name,
                "config": strategies,
                "changes": config.get("changes"),
                "timestamp": _format_timestamp(config.get("timestamp")),
                "tables": tables,
            }
        )
    return results


def render_site(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Render the about/results templates and write the static site to disk.

    Args:
        results (list[dict[str, Any]]): Per-result data as produced by discover_results.
        output_dir (Path): Directory to write the generated HTML files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    results_html = env.get_template("results.html").render(results=results)
    about_html = env.get_template("about.html").render()

    (output_dir / "results.html").write_text(results_html, encoding="utf-8")
    (output_dir / "about.html").write_text(about_html, encoding="utf-8")
    (output_dir / "index.html").write_text(about_html, encoding="utf-8")


if __name__ == "__main__":
    results = discover_results(RESULTS_DIR)
    render_site(results, DIST_DIR)
    print(f"Rendered {len(results)} result(s) to {DIST_DIR}/")
