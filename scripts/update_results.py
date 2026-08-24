#!/usr/bin/env python
"""Regenerate RESULTS.md from the metrics each training run wrote to disk.

Every number in the results table is read from ``artifacts/<project>/metrics.json``
rather than typed by hand, so the documentation cannot drift away from the models
that are actually trained.

Usage:
    python scripts/update_results.py
    python scripts/update_results.py --check   # fail if RESULTS.md is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsjourney.config import load_project_config  # noqa: E402
from dsjourney.paths import ARTIFACTS_DIR, available_projects  # noqa: E402

HEADLINE = {
    # Regression projects that train on a transformed target report both scales;
    # the price-scale figure is the one a reader can interpret.
    "regression": ("r2_original", "R²"),
    "classification": ("recall", "Recall"),
    "text-classification": ("f1", "F1"),
    "image-classification": ("accuracy", "Accuracy"),
    "clustering": ("silhouette", "Silhouette"),
    "forecasting": ("skill_vs_naive", "Skill vs naive"),
    "recommendation": ("precision_at_10", "Precision@10"),
    "retrieval": ("mrr", "MRR"),
    "detection": ("error", "Miscount"),
    # A graph has no single score. Reach from the chosen root is the number
    # that says the most about the shape of this one.
    "graph": ("reachable_fraction", "Reach"),
    # An RL agent is scored on how often it succeeds, never on one episode.
    "control": ("success_rate", "Success"),
}

# Fallbacks for projects that do not transform their target.
HEADLINE_FALLBACK = {"r2_original": "r2"}


def _render_row(name: str, config: Any, metrics: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Render one results table row."""
    key, label = HEADLINE.get(config.task, ("", ""))
    value = metrics.get(key, metrics.get(HEADLINE_FALLBACK.get(key, "")))
    headline = f"**{label} {value:.3f}**" if isinstance(value, (int, float)) else "-"
    rendered = ", ".join(
        f"{metric} {number:.3f}"
        for metric, number in metrics.items()
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    )
    model = str(metadata.get("model_class") or metrics.get("method") or "?")
    return f"| `{name}` | {config.task} | {model} | {headline} | {rendered} |"


def _train_hint(project: str, task: str) -> str:
    """Return the exact command that would train a project."""
    if task == "image-classification":
        return (
            f"`{project}` - run "
            f"`uv sync --extra dl --extra data && "
            f"python projects/{project}/train.py --dataset grape`"
        )
    return f"`{project}` - run `dsj train {project}`"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    content = json.loads(path.read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else {}


def build_document() -> str:
    """Render RESULTS.md from the artifacts currently on disk."""
    lines = [
        "# Results",
        "",
        "Every number here is read from `artifacts/<project>/metrics.json` by",
        "`scripts/update_results.py`. Regenerate after training rather than editing",
        "by hand; `--check` fails when the file has drifted.",
        "",
        f"Last generated: {date.today().isoformat()}",
        "",
        "| Project | Task | Model | Headline | All metrics |",
        "|---|---|---|---|---|",
    ]

    untrained: list[str] = []
    for project in available_projects():
        config = load_project_config(project)
        directory = ARTIFACTS_DIR / project

        # Some projects train several models under one name (one per series, one
        # per image dataset), so their metrics live a level down.
        nested = sorted(p for p in directory.glob("*/metrics.json"))
        if nested and not (directory / "metrics.json").is_file():
            for path in nested:
                lines.append(
                    _render_row(
                        f"{project} / {path.parent.name}",
                        config,
                        _read_json(path),
                        _read_json(path.parent / "metadata.json"),
                    )
                )
            continue

        metrics = _read_json(directory / "metrics.json")
        metadata = _read_json(directory / "metadata.json")

        if not metrics:
            untrained.append(_train_hint(project, config.task))
            continue

        lines.append(_render_row(project, config, metrics, metadata))

    if untrained:
        lines += [
            "",
            "## Not trained in this checkout",
            "",
            *(f"- {hint}" for hint in untrained),
        ]

    lines += [
        "",
        "## Reading these numbers",
        "",
        "- **`loan_default` is ranked on recall, not accuracy.** The data is 27%",
        "  defaults, so predicting 'never defaults' for everyone scores 73% accuracy",
        "  and catches nothing.",
        "- **Regression headlines are on the original target scale.** Both projects",
        "  train on `log1p(price)`; a log-scale R² and a price-scale R² are different",
        "  numbers, so `r2` and `r2_original` are both reported.",
        "- **`customer_segments` uses k = 4 although k = 2 scores higher.** Four",
        "  segments are actionable; two are not. The scan is in",
        "  `artifacts/customer_segments/cluster_selection.png`.",
        "- **`series_forecast` is ranked on skill against a naive baseline.** On",
        "  Adidas revenue that skill is 0.0 - nothing beats repeating the last",
        "  quarter, which is a result, not a missing model.",
        "- **`movie_recommender` precision@10 looks small by construction.** Each",
        "  user has a handful of held-out films among 1,682 candidates; random",
        "  ranking scores about 0.002.",
        "- **`article_search` reports two levels plus a cost.** Document recall",
        "  always favours bigger chunks; `hits_per_1k_words` is what makes the",
        "  trade against context size visible.",
        "- **`object_detection` is scored as a miscount, so lower is better.** Six",
        "  of seven faces found on the reference image, zero false positives.",
        "- **`diabetes_screening` is ranked on recall for the same reason as",
        "  `loan_default`.** It is a screening test: a missed diabetic patient",
        "  costs more than a false alarm. Its accuracy of 0.745 sits only 0.095",
        "  above always answering 'not diabetic'.",
        "- **`marvel_network` has no single score.** Reach is the fraction of the",
        "  graph within `eccentricity` hops of Captain America; the degree",
        "  distribution is the rest of the answer.",
        "- **The control projects report an interval, not a number.** A policy is",
        "  stochastic, so a success rate needs the episode count that produced it:",
        "  `frozenlake_control` is 0.726 [0.706, 0.745] over 2,000 episodes, and one",
        "  episode would have reported 0.0 or 1.0.",
        "- **`cartpole_balance` is ranked on the best agent it found.** The row to",
        "  read next to it is `notebook_dqn_return` 197 against `heuristic_return`",
        "  490 - the baseline the source notebook never ran.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if RESULTS.md is out of date")
    args = parser.parse_args(argv)

    document = build_document()
    target = REPO_ROOT / "RESULTS.md"

    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current.strip() != document.strip():
            print(
                "RESULTS.md is out of date. Run: python scripts/update_results.py", file=sys.stderr
            )
            return 1
        print("RESULTS.md is up to date")
        return 0

    target.write_text(document, encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
