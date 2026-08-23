#!/usr/bin/env python
"""Train the customer segmentation model.

Equivalent to ``dsj train customer_segments``; kept as a standalone entry point
so the project can be run, scheduled or containerised on its own.

Usage:
    python projects/customer_segments/train.py
    python projects/customer_segments/train.py --max-k 12
"""

from __future__ import annotations

import argparse

from dsjourney.training import train_clustering
from projects.customer_segments import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-k", type=int, default=10, help="largest cluster count to scan")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    raw = pipeline.load_raw()
    rfm = pipeline.build_rfm(raw)
    features = pipeline.log_scale_rfm(rfm)

    report = train_clustering(
        pipeline.CONFIG, features, k_range=range(2, args.max_k + 1), save=not args.no_save
    )

    print(report.summary())
    print()
    print(pipeline.describe_segments(rfm, report.predictions).to_string())
    if report.artifacts_dir:
        print(f"\nartifacts: {report.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
