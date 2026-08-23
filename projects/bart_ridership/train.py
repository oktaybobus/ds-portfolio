#!/usr/bin/env python
"""Train the BART ridership demand model on a chronological split.

The model is fitted on 2016 and scored on 2017. ``--compare-random`` also fits
the same model on a shuffled split of the same rows, which is what the source
notebook did, so the size of the difference is visible rather than assumed.

Usage:
    python projects/bart_ridership/train.py
    python projects/bart_ridership/train.py --sample 1000000
    python projects/bart_ridership/train.py --compare-random
"""

from __future__ import annotations

import argparse

from dsjourney import evaluate, viz
from dsjourney.artifacts import ModelBundle, save_bundle
from dsjourney.benchmark import build_model
from dsjourney.training import train_supervised
from projects.bart_ridership import pipeline

TARGET = "throughput_log"


def train_chronological(sample: int | None, *, save: bool = True) -> dict[str, float]:
    """Fit on 2016, score on 2017, and persist the bundle."""
    train_frame, test_frame = pipeline.chronological_frames(sample)
    print(f"train (2016): {len(train_frame):,} rows | test (2017): {len(test_frame):,} rows")

    features = [c for c in train_frame.columns if c != TARGET]
    model = build_model(
        "regression", pipeline.CONFIG.model.estimator, **pipeline.CONFIG.model.params
    )
    model.fit(train_frame[features], train_frame[TARGET])

    predictions = model.predict(test_frame[features])
    metrics = evaluate.regression_scores(test_frame[TARGET], predictions)
    metrics |= {
        f"{name}_original": value
        for name, value in evaluate.regression_scores(
            pipeline.postprocess(test_frame[TARGET].to_numpy()),
            pipeline.postprocess(predictions),
        ).items()
    }
    print(" | ".join(f"{name} {value:.4f}" for name, value in metrics.items()))

    if save:
        bundle = ModelBundle(
            project=pipeline.CONFIG.name,
            task="regression",
            model=model,
            feature_names=features,
            metrics=metrics,
            extra={
                "estimator_key": pipeline.CONFIG.model.estimator,
                "selected_by": "config",
                "split": "chronological (train 2016, test 2017)",
                "target": TARGET,
                "train_rows": len(train_frame),
                "test_rows": len(test_frame),
            },
        )
        directory = save_bundle(bundle)
        viz.save_figure(
            viz.residual_plot(test_frame[TARGET], predictions), directory / "residuals.png"
        )
        print(f"artifacts: {directory}")
    return metrics


def compare_with_random_split(sample: int | None) -> None:
    """Score the same model on a shuffled split, the way the notebook did.

    Both halves then contain the same station pairs at the same hours in the
    same seasons, so the model is effectively looking up rows it has already
    seen. The gap between the two numbers is the size of that illusion.
    """
    import pandas as pd

    train_frame, test_frame = pipeline.chronological_frames(sample)
    combined = pd.concat([train_frame, test_frame], ignore_index=True)

    report = train_supervised(
        pipeline.CONFIG,
        combined,
        save=False,
        make_plots=False,
        inverse_transform=pipeline.postprocess,
    )
    print("\nrandom split (what the notebook measured):")
    print(" | ".join(f"{k} {v:.4f}" for k, v in report.bundle.metrics.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=pipeline.DEFAULT_SAMPLE,
        help="rows per year; 0 reads all 13.3 million",
    )
    parser.add_argument("--compare-random", action="store_true", help="also score a shuffled split")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    sample = None if args.sample == 0 else args.sample
    metrics = train_chronological(sample, save=not args.no_save)

    if args.compare_random:
        compare_with_random_split(sample)
        print(
            "\nThe chronological score is the one that would survive deployment: "
            f"R2 {metrics['r2']:.4f} on a year the model never saw."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
