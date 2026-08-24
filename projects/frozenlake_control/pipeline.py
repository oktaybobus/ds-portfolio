"""The FrozenLake environment, as data.

A control project has no CSV. What it has is a transition table: for every
state and action, where the agent lands and with what probability.
:func:`load_raw` returns that table so the generic commands - ``dsj info``,
``dsj eda-report`` - still describe the thing the project is about.

Reading it is also what makes the comparison in ``train.py`` possible. Value
iteration solves the environment exactly *from* this table, which gives
Q-learning something to be judged against. The notebook had no such reference,
so the only available verdict on its agent was "it ran".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import rl
from dsjourney.config import load_project_config

CONFIG = load_project_config("frozenlake_control")

ENV_ID = "FrozenLake-v1"

# The 4x4 map, as gymnasium lays it out. S start, F frozen, H hole, G goal.
MAP_4X4 = ("SFFF", "FHFH", "FFFH", "HFFG")


def env_kwargs(*, slippery: bool | None = None) -> dict[str, Any]:
    """Environment settings, with the config's slipperiness unless overridden."""
    configured = bool(CONFIG.model.params["is_slippery"])
    return {"is_slippery": configured if slippery is None else slippery}


def load_raw(*, slippery: bool | None = None) -> pd.DataFrame:
    """Return the transition table as one row per (state, action, outcome)."""
    env = rl.make_env(ENV_ID, **env_kwargs(slippery=slippery)).unwrapped
    rows = [
        {
            "state": state,
            "action": action,
            "probability": probability,
            "next_state": next_state,
            "reward": reward,
            "terminal": done,
        }
        for state in range(int(env.observation_space.n))  # type: ignore[attr-defined]
        for action in range(int(env.action_space.n))  # type: ignore[attr-defined]
        for probability, next_state, reward, done in env.P[state][action]  # type: ignore[attr-defined]
    ]
    env.close()
    return pd.DataFrame(rows)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the transition table unchanged; an agent learns by acting, not by fitting."""
    return frame


def tile_kinds() -> pd.Series:
    """Return the map tile of each state id, for reading a Q-table by hand."""
    return pd.Series([tile for row in MAP_4X4 for tile in row], name="tile")


def learning_config(**overrides: Any) -> rl.QLearningConfig:
    """Build the configured Q-learning settings, with any overrides applied."""
    params = CONFIG.model.params
    settings = {
        "episodes": int(params["episodes"]),
        "alpha": float(params["alpha"]),
        "gamma": float(params["gamma"]),
        "schedule": str(params["schedule"]),
        "exploration_fraction": float(params["exploration_fraction"]),
        "epsilon_start": float(params["epsilon_start"]),
        "epsilon_min": float(params["epsilon_min"]),
    }
    settings.update(overrides)
    return rl.QLearningConfig(**settings)  # type: ignore[arg-type]


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: an agent is asked for an action, not scored on a row."""
    raise NotImplementedError(
        "frozenlake_control trains an agent. Use: python projects/frozenlake_control/train.py"
    )
