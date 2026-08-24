#!/usr/bin/env python
"""Train and measure a FrozenLake agent.

Usage:
    python projects/frozenlake_control/train.py
    python projects/frozenlake_control/train.py --schedule geometric
    python projects/frozenlake_control/train.py --no-slippery --seeds 12
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from dsjourney import rl
from dsjourney.paths import project_artifacts_dir
from projects.frozenlake_control import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    params = pipeline.CONFIG.model.params
    parser.add_argument("--seeds", type=int, default=int(params["seeds"]))
    parser.add_argument("--episodes", type=int, default=int(params["episodes"]))
    parser.add_argument("--eval-episodes", type=int, default=int(params["eval_episodes"]))
    parser.add_argument(
        "--schedule",
        choices=("linear", "geometric"),
        default=str(params["schedule"]),
        help="geometric is the notebook's epsilon *= 0.995 per episode",
    )
    parser.add_argument(
        "--no-slippery",
        action="store_true",
        help="the notebook's setting, where the task is a shortest path rather than RL",
    )
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    if not rl.gymnasium_installed():
        print(rl.INSTALL_HINT)
        return 2

    slippery = not args.no_slippery
    kwargs = pipeline.env_kwargs(slippery=slippery)
    print(f"{pipeline.ENV_ID} | is_slippery={slippery} | {args.schedule} epsilon schedule")

    # The exact optimum, read from the transition table rather than sampled.
    optimal_q, values, sweeps = rl.value_iteration(pipeline.ENV_ID, env_kwargs=kwargs)
    optimal = rl.evaluate_policy(
        pipeline.ENV_ID,
        rl.greedy_policy(optimal_q),
        episodes=args.eval_episodes,
        seed=999,
        env_kwargs=kwargs,
    )
    print(f"\nvalue iteration ({sweeps} sweeps, V(start)={values[0]:.4f})")
    print(f"  {optimal.summary()}")

    chance = rl.evaluate_policy(
        pipeline.ENV_ID,
        rl.random_policy(pipeline.ENV_ID, env_kwargs=kwargs),
        episodes=args.eval_episodes,
        seed=999,
        env_kwargs=kwargs,
    )
    print(f"random policy\n  {chance.summary()}")

    config = pipeline.learning_config(episodes=args.episodes, schedule=args.schedule)
    print(f"\nQ-learning, {args.seeds} seeds x {args.episodes:,} episodes...")
    table = rl.compare_seeds(
        pipeline.ENV_ID,
        config,
        seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        env_kwargs=kwargs,
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rates = table["success_rate"].to_numpy()
    failed = int((rates < 0.05).sum())
    print(
        f"\nacross seeds: mean {rates.mean():.3f} | "
        f"min {rates.min():.3f} | max {rates.max():.3f} | "
        f"learned nothing on {failed}/{args.seeds}"
    )
    if failed:
        print(
            f"  A single run of this configuration reports success "
            f"{1 - failed / args.seeds:.0%} of the time and total failure the rest. "
            f"The notebook ran it once."
        )

    best = float(rates.max())
    print(
        f"\none episode is not an evaluation: at a true rate of {optimal.success_rate:.3f}, "
        f"pinning it to +/-0.02 takes {rl.episodes_for_precision(optimal.success_rate):,} "
        f"episodes"
    )

    metrics = {
        "success_rate": float(np.median(rates)),
        "ci_low": float(table["ci_low"].median()),
        "ci_high": float(table["ci_high"].median()),
        "mean_return": float(table["mean_return"].median()),
        "optimal_success_rate": optimal.success_rate,
        "gap_to_optimal": optimal.success_rate - float(np.median(rates)),
        "seeds_failed": float(failed),
    }
    print("\n" + " | ".join(f"{name} {value:.4g}" for name, value in metrics.items()))

    if not args.no_save:
        directory = project_artifacts_dir("frozenlake_control", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "TabularQLearning",
            "environment": pipeline.ENV_ID,
            "is_slippery": slippery,
            "schedule": args.schedule,
            "episodes": args.episodes,
            "seeds": args.seeds,
            "eval_episodes": args.eval_episodes,
            "value_iteration_sweeps": sweeps,
            "best_seed_success_rate": best,
            "random_success_rate": chance.success_rate,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        table.to_csv(directory / "seed_sweep.csv", index=False)
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
