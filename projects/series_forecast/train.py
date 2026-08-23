#!/usr/bin/env python
"""Fit and score forecasters on a chronological holdout.

Usage:
    python projects/series_forecast/train.py
    python projects/series_forecast/train.py --series adidas_revenue
    python projects/series_forecast/train.py --all
"""

from __future__ import annotations

import argparse
import json

from dsjourney import forecasting, viz
from dsjourney.paths import project_artifacts_dir
from projects.series_forecast import pipeline


def run(series: str, *, horizon: int | None = None, save: bool = True) -> dict[str, float]:
    """Score every forecaster on one series and persist the winner's metrics."""
    spec = pipeline.spec_for(series)
    values = pipeline.build_series(pipeline.load_raw(series), spec)
    split = forecasting.chronological_split(values, horizon=horizon or spec.horizon)

    print(f"[{series}] {len(values)} observations, {split.horizon}-period holdout")
    strength = forecasting.seasonal_strength(split.train, period=spec.period)
    print(
        f"[{series}] trend strength {strength['trend_strength']:.3f}, "
        f"seasonal strength {strength['seasonal_strength']:.3f}"
    )

    table = forecasting.compare_forecasters(
        split,
        params={
            "seasonal_naive": {"period": spec.period},
            "holt_winters": {"period": spec.period},
            "sarima": {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, min(spec.period, 12))},
        },
    )
    print(table.to_string(index=False))

    best = table.iloc[0]
    winner = str(best["method"])
    metrics = {
        key: float(best[key])
        for key in ("mae", "rmse", "mape", "mase", "skill_vs_naive")
        if key in best
    }
    skill = metrics.get("skill_vs_naive", float("nan"))
    print(
        f"[{series}] best: {winner} - MAE {metrics['mae']:.3f} {spec.units}, "
        f"{skill:+.1%} against the naive baseline"
    )

    if save:
        _save(series, winner, metrics, strength, table, split, spec)
    return metrics


def _save(series, winner, metrics, strength, table, split, spec) -> None:  # type: ignore[no-untyped-def]
    """Write metrics, the comparison table and a forecast plot."""
    directory = project_artifacts_dir("series_forecast", create=True) / series
    directory.mkdir(parents=True, exist_ok=True)

    # Metrics and provenance are kept apart: RESULTS.md renders every number in
    # metrics.json as a score, so observation counts do not belong there.
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "model_class": winner,
        "series": spec.title,
        "units": spec.units,
        "observations": int(len(split.train) + len(split.test)),
        "horizon": split.horizon,
        **strength,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    table.to_csv(directory / "comparison.csv", index=False)

    forecast = forecasting.build_forecast(
        winner,
        split.train,
        split.horizon,
        **({"period": spec.period} if winner in {"seasonal_naive", "holt_winters"} else {}),
    )
    viz.save_figure(
        viz.forecast_plot(split.train, split.test, forecast, title=f"{spec.title} - {winner}"),
        directory / "forecast.png",
    )
    print(f"[{series}] artifacts: {directory}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series", choices=sorted(pipeline.SERIES), default=pipeline.DEFAULT_SERIES
    )
    parser.add_argument("--all", action="store_true", help="run every series in the catalogue")
    parser.add_argument("--horizon", type=int, help="override the holdout length")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    targets = sorted(pipeline.SERIES) if args.all else [args.series]
    for series in targets:
        run(series, horizon=args.horizon, save=not args.no_save)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
