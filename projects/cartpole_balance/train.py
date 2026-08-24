#!/usr/bin/env python
"""Train a CartPole DQN and put it next to the baselines.

Usage:
    python projects/cartpole_balance/train.py
    python projects/cartpole_balance/train.py --timesteps 20000
    python projects/cartpole_balance/train.py --skip-dqn
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from dsjourney import rl
from dsjourney.paths import project_artifacts_dir
from projects.cartpole_balance import pipeline

# The notebook's hyper-parameters, verbatim.
NOTEBOOK_DQN = {
    "learning_rate": 1e-3,
    "buffer_size": 50_000,
    "learning_starts": 1000,
    "batch_size": 64,
    "gamma": 0.99,
}

# The RL Baselines3 Zoo configuration for CartPole, on the same timestep budget.
# The difference that matters is train_freq/gradient_steps: the defaults take
# one gradient step every four environment steps, this takes 128 every 256 -
# about eight times the learning from the same experience - on a wider network.
TUNED_DQN = {
    "learning_rate": 2.3e-3,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "target_update_interval": 10,
    "train_freq": 256,
    "gradient_steps": 128,
    "exploration_fraction": 0.16,
    "exploration_final_eps": 0.04,
    "policy_kwargs": {"net_arch": [256, 256]},
}


def train_dqn(settings: dict[str, Any], *, timesteps: int, seed: int) -> tuple[rl.Policy, float]:
    """Fit a DQN and return a greedy policy plus the seconds it took."""
    from stable_baselines3 import DQN

    env = rl.make_env(pipeline.ENV_ID)
    started = time.perf_counter()
    model = DQN("MlpPolicy", env, verbose=0, seed=seed, **settings)
    model.learn(total_timesteps=timesteps)
    elapsed = time.perf_counter() - started
    env.close()

    def act(observation: Any) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    return act, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    params = pipeline.CONFIG.model.params
    parser.add_argument("--timesteps", type=int, default=int(params["total_timesteps"]))
    parser.add_argument("--episodes", type=int, default=int(params["eval_episodes"]))
    parser.add_argument("--seed", type=int, default=int(params["seed"]))
    parser.add_argument(
        "--skip-dqn", action="store_true", help="baselines only; no stable-baselines3 needed"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    if not rl.gymnasium_installed():
        print(rl.INSTALL_HINT)
        return 2

    success = float(params["success_return"])
    scored: dict[str, rl.EvaluationResult] = {}
    seconds: dict[str, float] = {}

    def score(name: str, policy: rl.Policy) -> None:
        scored[name] = rl.evaluate_policy(
            pipeline.ENV_ID,
            policy,
            episodes=args.episodes,
            seed=7,
            success_return=success,
        )

    print(
        f"{pipeline.ENV_ID} | solved at {success:.0f} of a maximum 500 | {args.episodes} episodes"
    )

    score("random", rl.random_policy(pipeline.ENV_ID, seed=args.seed))
    score("heuristic", pipeline.heuristic_policy())

    if not args.skip_dqn:
        if not rl.sb3_installed():
            print(f"\nstable-baselines3 is not installed.\n{rl.INSTALL_HINT}")
            return 2
        for name, settings in (("dqn_notebook", NOTEBOOK_DQN), ("dqn_tuned", TUNED_DQN)):
            print(f"\ntraining {name} for {args.timesteps:,} timesteps...")
            policy, elapsed = train_dqn(settings, timesteps=args.timesteps, seed=args.seed)
            seconds[name] = elapsed
            print(f"  {elapsed:.1f}s")
            score(name, policy)

    print(f"\n{'agent':14} {'mean return':>12} {'solved':>8}  95% interval")
    for name, result in scored.items():
        print(
            f"{name:14} {result.mean_return:12.1f} {result.success_rate:8.1%}"
            f"  [{result.ci_low:.3f}, {result.ci_high:.3f}]"
        )

    heuristic = scored["heuristic"]
    if "dqn_notebook" in scored:
        notebook = scored["dqn_notebook"]
        ratio = heuristic.mean_return / max(notebook.mean_return, 1e-9)
        print(
            f"\n  The notebook trained the DQN and never scored it. It reaches "
            f"{notebook.mean_return:.0f} of 500 and solves the environment "
            f"{notebook.success_rate:.0%} of the time.\n"
            f"  The two-line heuristic reaches {heuristic.mean_return:.0f} - "
            f"{ratio:.1f}x more - with no training at all."
        )
    if "dqn_tuned" in scored:
        tuned = scored["dqn_tuned"]
        print(
            f"  The same algorithm on the same budget, tuned, reaches "
            f"{tuned.mean_return:.0f} and solves it {tuned.success_rate:.0%} of the time. "
            f"The gap was hyper-parameters, and only measurement shows that."
        )

    best = max(scored.values(), key=lambda r: r.mean_return)
    metrics = {
        "mean_return": best.mean_return,
        "success_rate": best.success_rate,
        "ci_low": best.ci_low,
        "ci_high": best.ci_high,
        "heuristic_return": heuristic.mean_return,
        "notebook_dqn_return": scored["dqn_notebook"].mean_return
        if "dqn_notebook" in scored
        else 0.0,
        "random_return": scored["random"].mean_return,
    }
    print("\n" + " | ".join(f"{name} {value:.4g}" for name, value in metrics.items()))

    if not args.no_save:
        directory = project_artifacts_dir("cartpole_balance", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "DQN" if "dqn_tuned" in scored else "Heuristic",
            "environment": pipeline.ENV_ID,
            "total_timesteps": args.timesteps,
            "eval_episodes": args.episodes,
            "seed": args.seed,
            "train_seconds": {k: round(v, 1) for k, v in seconds.items()},
            "agents": {name: result.as_metrics() for name, result in scored.items()},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
