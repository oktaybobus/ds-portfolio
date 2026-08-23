#!/usr/bin/env python
"""Train the loan default model.

Equivalent to ``dsj train loan_default``; kept as a standalone entry point so the
project can be run, scheduled or containerised on its own.

Usage:
    python projects/loan_default/train.py
    python projects/loan_default/train.py --benchmark
"""

from __future__ import annotations

import argparse

from dsjourney.training import train_supervised
from projects.loan_default import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", action="store_true", help="sweep every estimator, keep the winner"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    frame = pipeline.build_features(pipeline.load_raw())
    report = train_supervised(
        pipeline.CONFIG, frame, benchmark=args.benchmark, save=not args.no_save
    )

    print(report.summary())
    if report.benchmark is not None:
        print(report.benchmark.table.to_string(index=False))
    if report.artifacts_dir:
        print(f"artifacts: {report.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
