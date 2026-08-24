"""Reinforcement learning, with the measurement the notebooks left out.

The source notebook trained a Q-table for 20,000 episodes, saved it, printed
"Training completed and model saved", and then checked it by running **one**
episode. On the deterministic FrozenLake that reports 1.0 or 0.0 and nothing in
between. On the slippery one - the version where the problem is actually a
reinforcement-learning problem - the optimal policy still fails 26% of the time,
so a single-episode check calls the best possible agent a failure in roughly one
run out of three.

That is the defect this module exists to prevent. :func:`evaluate_policy` never
returns a bare number: it returns a rate with a confidence interval and the
episode count that produced it, and :func:`episodes_for_precision` says how many
episodes a given precision needs before the run starts.

Two smaller things are fixed by construction:

* **Bootstrapping through a terminal state.** The notebook's update adds
  ``gamma * max(Q[next_state])`` whatever happened, but a terminated episode has
  no future to discount - only a *truncated* one does. It collapses ``terminated``
  and ``truncated`` into one ``done`` flag and loses the distinction.
* **``argmax`` over an all-zero row is not a decision.** Every state the agent
  never visited keeps a row of zeros, and ``np.argmax`` answers 0 - a real
  action, chosen by tie-breaking rather than by learning.
  :func:`undecided_states` counts them, excluding the terminal states where a
  zero row is correct.

Nothing here renders. The toy-text environments open a pygame window for
``render_mode="human"``, which is the same defect :mod:`dsjourney.detection`
documents for ``cv2.imshow``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gymnasium import Env

# A policy maps an observation to an action. Tabular agents take an int state;
# the CartPole agents take an array. Both fit this signature.
Policy = Callable[[Any], int]

INSTALL_HINT = (
    "Reinforcement learning needs the optional extra:\n"
    "  uv sync --extra rl        # gymnasium, for the tabular projects\n"
    "  uv sync --extra deeprl    # adds stable-baselines3 and torch"
)


def gymnasium_installed() -> bool:
    """True when the ``rl`` extra is installed."""
    import importlib.util

    return importlib.util.find_spec("gymnasium") is not None


def sb3_installed() -> bool:
    """True when the ``deeprl`` extra is installed."""
    import importlib.util

    return importlib.util.find_spec("stable_baselines3") is not None


def require_gymnasium() -> None:
    """Raise with installation instructions when gymnasium is missing."""
    if not gymnasium_installed():
        raise ImportError(f"gymnasium is not installed.\n{INSTALL_HINT}")


def make_env(env_id: str, **kwargs: Any) -> Env:
    """Build an environment, never a window.

    ``render_mode`` is deliberately not passed through. A rendered toy-text
    environment opens pygame, which blocks on a machine with no display - the
    same failure ``cv2.imshow`` causes in :mod:`dsjourney.detection`.
    """
    require_gymnasium()
    import gymnasium as gym

    kwargs.pop("render_mode", None)
    return gym.make(env_id, **kwargs)


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a Wilson score interval for a success rate.

    The textbook ``p +/- 1.96 * sqrt(p(1-p)/n)`` misbehaves exactly where RL
    results live: at rates near 0 or 1 it produces bounds outside [0, 1], and on
    the small episode counts people actually run it is too narrow. Wilson stays
    inside the range and is honest at n = 20.
    """
    if trials <= 0:
        return 0.0, 0.0
    # 0.95 -> 1.96, 0.99 -> 2.576; the normal quantile via the inverse error function.
    z = math.sqrt(2.0) * _inverse_erf(confidence)
    phat = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _inverse_erf(confidence: float) -> float:
    """Inverse error function at ``confidence``, by Newton refinement."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    # Winitzki's approximation, then two Newton steps against math.erf.
    a = 0.147
    ln = math.log(1 - confidence**2)
    term = 2 / (math.pi * a) + ln / 2
    guess = math.copysign(math.sqrt(math.sqrt(term**2 - ln / a) - term), confidence)
    for _ in range(2):
        error = math.erf(guess) - confidence
        guess -= error / (2 / math.sqrt(math.pi) * math.exp(-(guess**2)))
    return guess


def episodes_for_precision(
    rate: float, *, half_width: float = 0.02, confidence: float = 0.95
) -> int:
    """Episodes needed to pin a success rate to +/- ``half_width``.

    Answers the question the notebook never asked. At a true rate of 0.74,
    resolving two percentage points takes about 1,850 episodes; one episode
    resolves nothing at all.
    """
    rate = min(max(rate, 0.0), 1.0)
    z = math.sqrt(2.0) * _inverse_erf(confidence)
    variance = max(rate * (1 - rate), 1e-9)
    return math.ceil(z**2 * variance / half_width**2)


@dataclass(frozen=True)
class EvaluationResult:
    """What a policy scored, and how sure we are of it."""

    episodes: int
    mean_return: float
    return_sd: float
    success_rate: float
    ci_low: float
    ci_high: float
    mean_length: float

    @property
    def ci_half_width(self) -> float:
        """Half the width of the confidence interval, in success-rate points."""
        return (self.ci_high - self.ci_low) / 2

    def summary(self) -> str:
        """One line, with the interval attached to the number."""
        return (
            f"return {self.mean_return:.2f} +/- {self.return_sd:.2f} | "
            f"success {self.success_rate:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}] over {self.episodes} episodes"
        )

    def as_metrics(self, prefix: str = "") -> dict[str, float]:
        """Flatten to a metrics dict for ``artifacts/<project>/metrics.json``."""
        return {
            f"{prefix}mean_return": self.mean_return,
            f"{prefix}success_rate": self.success_rate,
            f"{prefix}ci_low": self.ci_low,
            f"{prefix}ci_high": self.ci_high,
            f"{prefix}mean_length": self.mean_length,
            f"{prefix}episodes": float(self.episodes),
        }


def evaluate_policy(
    env_id: str,
    policy: Policy,
    *,
    episodes: int = 2000,
    seed: int = 0,
    success_return: float = 1.0,
    env_kwargs: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Run ``policy`` for ``episodes`` and report the result with its interval.

    ``success_return`` is the return at or above which an episode counts as a
    success: 1.0 on FrozenLake, where the goal pays exactly once, and 475 on
    CartPole, which is where that environment is defined as solved.
    """
    env = make_env(env_id, **(env_kwargs or {}))
    returns: list[float] = []
    lengths: list[int] = []
    env.reset(seed=seed)
    try:
        for _ in range(episodes):
            observation, _ = env.reset()
            total = 0.0
            steps = 0
            while True:
                observation, reward, terminated, truncated, _ = env.step(policy(observation))
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(steps)
    finally:
        env.close()

    array = np.asarray(returns, dtype=float)
    successes = int((array >= success_return).sum())
    low, high = wilson_interval(successes, episodes)
    return EvaluationResult(
        episodes=episodes,
        mean_return=float(array.mean()),
        return_sd=float(array.std()),
        success_rate=successes / episodes,
        ci_low=low,
        ci_high=high,
        mean_length=float(np.mean(lengths)),
    )


def greedy_policy(q_table: np.ndarray) -> Policy:
    """Return the policy that takes the best-known action in each state."""

    def act(state: Any) -> int:
        return int(np.argmax(q_table[int(state)]))

    return act


def random_policy(
    env_id: str, *, seed: int = 0, env_kwargs: dict[str, Any] | None = None
) -> Policy:
    """Return a uniformly random policy - the floor every agent must clear."""
    env = make_env(env_id, **(env_kwargs or {}))
    actions = int(env.action_space.n)  # type: ignore[attr-defined]
    env.close()
    generator = np.random.default_rng(seed)

    def act(observation: Any) -> int:  # noqa: ARG001 - a random policy ignores the state
        return int(generator.integers(0, actions))

    return act


def terminal_states(env_id: str, *, env_kwargs: dict[str, Any] | None = None) -> np.ndarray:
    """Return the absorbing states of a tabular environment.

    Needed to read :func:`undecided_states` honestly: a terminal state has an
    all-zero Q-row in *any* correct solution, including an exact one, so
    counting it as "never learned" overstates the problem by five states on
    FrozenLake.
    """
    env = make_env(env_id, **(env_kwargs or {})).unwrapped
    states = int(env.observation_space.n)  # type: ignore[attr-defined]
    actions = int(env.action_space.n)  # type: ignore[attr-defined]
    transitions = env.P  # type: ignore[attr-defined]
    absorbing = [
        state
        for state in range(states)
        if all(done for action in range(actions) for *_, done in transitions[state][action])
    ]
    env.close()
    return np.asarray(absorbing, dtype=int)


def undecided_states(q_table: np.ndarray, *, ignore: np.ndarray | None = None) -> np.ndarray:
    """Return the states whose Q-row is entirely zero.

    ``np.argmax`` on such a row returns action 0, which looks like a decision
    and is not one - the action was picked by tie-breaking, not by learning.
    Pass :func:`terminal_states` as ``ignore`` to exclude the states where an
    all-zero row is correct rather than untrained.
    """
    zero_rows = np.flatnonzero(~q_table.any(axis=1))
    if ignore is None:
        return zero_rows
    return np.setdiff1d(zero_rows, np.asarray(ignore, dtype=int))


@dataclass(frozen=True)
class QLearningConfig:
    """Hyper-parameters for tabular Q-learning.

    ``schedule`` is the one that decides whether the agent learns at all.
    ``"geometric"`` is the notebook's ``epsilon *= 0.995`` per episode, which
    reaches the floor after about 920 episodes however many were budgeted - so
    of 20,000 episodes, 19,000 are run at 1% exploration on whatever the table
    held at episode 920. ``"linear"`` spreads the decay over
    ``exploration_fraction`` of the run, so the budget actually buys something.
    """

    episodes: int = 20_000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    schedule: str = "linear"
    exploration_fraction: float = 0.5
    seed: int = 0
    curve_window: int = 500

    def epsilon_at(self, episode: int) -> float:
        """Exploration rate for a given episode under this schedule."""
        if self.schedule == "geometric":
            return max(self.epsilon_start * self.epsilon_decay**episode, self.epsilon_min)
        if self.schedule == "linear":
            span = max(self.exploration_fraction * self.episodes, 1.0)
            progress = min(episode / span, 1.0)
            return self.epsilon_start + progress * (self.epsilon_min - self.epsilon_start)
        raise ValueError(f"unknown schedule {self.schedule!r}; use 'geometric' or 'linear'")


@dataclass
class TrainingRun:
    """A trained Q-table alongside the curve that produced it."""

    q_table: np.ndarray
    curve: pd.DataFrame
    episodes: int

    @property
    def final_epsilon(self) -> float:
        """Exploration rate the run finished on."""
        return float(self.curve["epsilon"].iloc[-1]) if len(self.curve) else 0.0


def train_q_learning(
    env_id: str,
    config: QLearningConfig | None = None,
    *,
    env_kwargs: dict[str, Any] | None = None,
) -> TrainingRun:
    """Train a tabular Q-learning agent and record how it got there.

    The update bootstraps only when the episode was *truncated*. A terminated
    episode has no successor to discount, and writing ``gamma * max(Q[s'])``
    anyway is the notebook's version - harmless on FrozenLake only because
    terminal rows are never updated and so stay zero. It is not harmless in
    general, and nothing about the code says which case you are in.
    """
    settings = config or QLearningConfig()
    env = make_env(env_id, **(env_kwargs or {}))
    generator = np.random.default_rng(settings.seed)
    q_table = np.zeros((int(env.observation_space.n), int(env.action_space.n)))  # type: ignore[attr-defined]

    records: list[dict[str, float]] = []
    env.reset(seed=settings.seed)

    try:
        for episode in range(settings.episodes):
            epsilon = settings.epsilon_at(episode)
            state, _ = env.reset()
            total = 0.0
            steps = 0
            while True:
                if generator.random() < epsilon:
                    action = int(generator.integers(0, q_table.shape[1]))
                else:
                    action = int(np.argmax(q_table[int(state)]))

                next_state, reward, terminated, truncated, _ = env.step(action)
                future = 0.0 if terminated else float(np.max(q_table[int(next_state)]))
                q_table[int(state), action] += settings.alpha * (
                    float(reward) + settings.gamma * future - q_table[int(state), action]
                )
                state = next_state
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            records.append(
                {
                    "episode": float(episode),
                    "return": total,
                    "length": float(steps),
                    "epsilon": epsilon,
                }
            )
    finally:
        env.close()

    curve = pd.DataFrame.from_records(records)
    curve["rolling_return"] = curve["return"].rolling(settings.curve_window, min_periods=1).mean()
    return TrainingRun(q_table=q_table, curve=curve, episodes=settings.episodes)


def value_iteration(
    env_id: str,
    *,
    gamma: float = 0.99,
    theta: float = 1e-10,
    max_sweeps: int = 10_000,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve an environment exactly from its transition table.

    Returns ``(q_table, values, sweeps)``. This is the ceiling: it reads the
    dynamics straight out of ``env.P`` instead of sampling them, so whatever it
    scores is the best any policy can do. Q-learning is worth judging against
    this rather than against nothing, which is what the notebook did.
    """
    env = make_env(env_id, **(env_kwargs or {})).unwrapped
    states = int(env.observation_space.n)  # type: ignore[attr-defined]
    actions = int(env.action_space.n)  # type: ignore[attr-defined]
    transitions = env.P  # type: ignore[attr-defined]
    values = np.zeros(states)

    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):  # noqa: B007 - the count is the return value
        delta = 0.0
        for state in range(states):
            best = max(
                sum(
                    probability * (reward + gamma * values[next_state] * (not done))
                    for probability, next_state, reward, done in transitions[state][action]
                )
                for action in range(actions)
            )
            delta = max(delta, abs(best - values[state]))
            values[state] = best
        if delta < theta:
            break

    q_table = np.zeros((states, actions))
    for state in range(states):
        for action in range(actions):
            q_table[state, action] = sum(
                probability * (reward + gamma * values[next_state] * (not done))
                for probability, next_state, reward, done in transitions[state][action]
            )
    env.close()
    return q_table, values, sweeps


def compare_seeds(
    env_id: str,
    config: QLearningConfig,
    *,
    seeds: int = 10,
    eval_episodes: int = 1000,
    eval_seed: int = 999,
    env_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train once per seed and return every run's score, not the best one.

    The whole point. A single training run of a stochastic algorithm is one
    sample from a distribution, and on the notebook's schedule that distribution
    has mass at both 0.0 and 0.7 - so "it works" and "it learns nothing" are
    both true reports of the same code. One row per seed makes that visible.
    """
    rows: list[dict[str, float]] = []
    for seed in range(seeds):
        run = train_q_learning(
            env_id, QLearningConfig(**{**config.__dict__, "seed": seed}), env_kwargs=env_kwargs
        )
        result = evaluate_policy(
            env_id,
            greedy_policy(run.q_table),
            episodes=eval_episodes,
            seed=eval_seed,
            env_kwargs=env_kwargs,
        )
        untrained = undecided_states(
            run.q_table, ignore=terminal_states(env_id, env_kwargs=env_kwargs)
        )
        rows.append(
            {
                "seed": float(seed),
                "success_rate": result.success_rate,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "mean_return": result.mean_return,
                "goals_during_training": float(run.curve["return"].sum()),
                "undecided_states": float(len(untrained)),
            }
        )
    return pd.DataFrame.from_records(rows)


def available_models() -> list[str]:
    """Return the algorithms a project config may name."""
    return ["q_learning", "value_iteration", "dqn"]
