#!/usr/bin/env python
"""Train the restaurant review sentiment model.

Equivalent to ``dsj train review_sentiment``.

Usage:
    python projects/review_sentiment/train.py
    python projects/review_sentiment/train.py --max-features 20000
"""

from __future__ import annotations

import argparse

from dsjourney.text import top_features
from dsjourney.training import train_text_classifier
from projects.review_sentiment import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF vocabulary cap")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    frame = pipeline.build_features(pipeline.load_raw())
    report = train_text_classifier(
        pipeline.CONFIG,
        frame[pipeline.TEXT_COLUMN],
        frame[pipeline.LABEL_COLUMN],
        save=not args.no_save,
        max_features=args.max_features,
    )

    print(report.summary())
    print()
    print("Most influential terms:")
    print(top_features(report.bundle.model, limit=12).to_string(index=False))
    if report.artifacts_dir:
        print(f"\nartifacts: {report.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
