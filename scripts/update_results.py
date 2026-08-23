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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsjourney.config import load_project_config  # noqa: E402
from dsjourney.paths import ARTIFACTS_DIR, available_projects  # noqa: E402

HEADLINE = {
    "regression": ("r2", "R²"),
    "classification": ("recall", "Recall"),
    "text-classification": ("f1", "F1"),
    "image-classification": ("accuracy", "Accuracy"),
    "clustering": ("silhouette", "Silhouette"),
}


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
        metrics = _read_json(directory / "metrics.json")
        metadata = _read_json(directory / "metadata.json")

        if not metrics:
            untrained.append(_train_hint(project, config.task))
            continue

        key, label = HEADLINE.get(config.task, ("", ""))
        value = metrics.get(key)
        headline = f"**{label} {value:.3f}**" if isinstance(value, (int, float)) else "-"
        rendered = ", ".join(
            f"{name} {number:.3f}"
            for name, number in metrics.items()
            if isinstance(number, (int, float))
        )
        model = str(metadata.get("model_class", "?"))
        lines.append(f"| `{project}` | {config.task} | {model} | {headline} | {rendered} |")

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
        "- **`laptop_price` metrics are on the log-transformed target.** MAPE is on",
        "  the same scale; the demo app converts predictions back to currency.",
        "- **`customer_segments` uses k = 4 although k = 2 scores higher.** Four",
        "  segments are actionable; two are not. The scan is in",
        "  `artifacts/customer_segments/cluster_selection.png`.",
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
