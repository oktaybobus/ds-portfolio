"""The CartPole environment and the baselines worth beating.

CartPole has no transition table to read - its state is four continuous
numbers - so there is nothing here to load. What there is instead is
:func:`heuristic_policy`, and it is the most useful thing in the project: an
agent that does not clear it has not earned the 50,000 timesteps it cost.

The source notebook trained a DQN and never compared it to anything, which is
how an agent scoring 197 out of 500 gets saved to disk and called done.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsjourney import rl
from dsjourney.config import load_project_config

CONFIG = load_project_config("cartpole_balance")

ENV_ID = "CartPole-v1"

# The four observations, in the order gymnasium returns them.
OBSERVATIONS = ("cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity")


def load_raw() -> pd.DataFrame:
    """Describe the observation space - the nearest thing this project has to data."""
    env = rl.make_env(ENV_ID)
    low, high = env.observation_space.low, env.observation_space.high  # type: ignore[attr-defined]
    env.close()
    return pd.DataFrame(
        {
            "observation": list(OBSERVATIONS),
            "index": range(len(OBSERVATIONS)),
            "low": np.asarray(low, dtype=float),
            "high": np.asarray(high, dtype=float),
        }
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the space description unchanged; the agent learns from interaction."""
    return frame


def heuristic_policy() -> rl.Policy:
    """Push the cart the way the pole is falling.

    ``pole_angle + pole_angular_velocity > 0`` means the pole is to the right or
    heading there, so move right. That is the whole policy: no training, no
    parameters, no replay buffer. It scores 490 of a possible 500.
    """

    def act(observation: Any) -> int:
        return int(observation[2] + observation[3] > 0)

    return act


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: an agent is asked for an action, not scored on a row."""
    raise NotImplementedError(
        "cartpole_balance trains an agent. Use: python projects/cartpole_balance/train.py"
    )
