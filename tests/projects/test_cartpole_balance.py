"""Tests for the CartPole project.

The heuristic is a pure function of the observation, so most of what matters
here is testable without stepping an environment at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsjourney import rl
from projects.cartpole_balance import pipeline

RANDOM_RETURN = 22.0
HEURISTIC_RETURN = 490.0
# The notebook configuration measured across six seeds. The tuned one is not
# pinned to a number anywhere, because it does not have a stable one.
NOTEBOOK_DQN_RANGE = (125.0, 235.0)


def test_the_heuristic_pushes_the_way_the_pole_leans() -> None:
    policy = pipeline.heuristic_policy()
    # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    assert policy(np.array([0.0, 0.0, 0.05, 0.0])) == 1, "pole right of centre -> push right"
    assert policy(np.array([0.0, 0.0, -0.05, 0.0])) == 0, "pole left of centre -> push left"
    # Angular velocity can outvote a small angle: the pole is heading back.
    assert policy(np.array([0.0, 0.0, 0.01, -0.5])) == 0


def test_the_heuristic_ignores_the_cart_entirely() -> None:
    """Two observations of four are unused, which is what makes it two lines."""
    policy = pipeline.heuristic_policy()
    for position, velocity in ((-2.0, -1.0), (0.0, 0.0), (2.0, 1.0)):
        assert policy(np.array([position, velocity, 0.05, 0.0])) == 1


def test_prepare_input_explains_the_right_entry_point() -> None:
    with pytest.raises(NotImplementedError, match=r"train\.py"):
        pipeline.prepare_input({})


def test_the_configured_success_threshold_is_the_published_one() -> None:
    """CartPole-v1 caps at 500 steps and is defined as solved at 475."""
    assert float(pipeline.CONFIG.model.params["success_return"]) == 475.0


@pytest.mark.needs_rl
def test_load_raw_describes_the_observation_space() -> None:
    frame = pipeline.load_raw()
    assert list(frame["observation"]) == list(pipeline.OBSERVATIONS)
    assert len(frame) == 4
    assert (frame["high"] > frame["low"]).all()


@pytest.mark.needs_rl
@pytest.mark.slow
def test_the_heuristic_solves_what_random_cannot() -> None:
    """The baseline the notebook's DQN needed to clear and never did."""
    chance = rl.evaluate_policy(
        pipeline.ENV_ID, rl.random_policy(pipeline.ENV_ID, seed=0), episodes=50, seed=7
    )
    heuristic = rl.evaluate_policy(
        pipeline.ENV_ID,
        pipeline.heuristic_policy(),
        episodes=50,
        seed=7,
        success_return=475.0,
    )
    assert chance.mean_return < 40
    assert heuristic.mean_return > 400
    assert heuristic.success_rate > 0.7


@pytest.mark.needs_deeprl
@pytest.mark.slow
def test_the_notebook_hyperparameters_lose_to_two_lines_of_physics() -> None:
    """The finding: 50,000 timesteps of DQN, unmeasured, beaten by a rule of thumb.

    Two seeds rather than one, because the claim being pinned is about the
    configuration and not about a run. This configuration is the reliable half
    of the result: across six seeds locally and one CI platform it lands
    between 126 and 233 and never solves the environment.

    There is deliberately no companion test asserting the *tuned* configuration
    succeeds. It reaches 500 on most seeds, collapsed to 105 on one of six, and
    scored below random on the CI runner - so such a test would be flaky, which
    would be the defect this project documents rather than a check against it.
    """
    from projects.cartpole_balance.train import NOTEBOOK_DQN, train_dqn

    heuristic = rl.evaluate_policy(
        pipeline.ENV_ID, pipeline.heuristic_policy(), episodes=50, seed=7, success_return=475.0
    )
    for seed in (0, 1):
        policy, _ = train_dqn(NOTEBOOK_DQN, timesteps=50_000, seed=seed)
        dqn = rl.evaluate_policy(pipeline.ENV_ID, policy, episodes=50, seed=7, success_return=475.0)
        assert dqn.mean_return < heuristic.mean_return, f"seed {seed}"
        assert dqn.success_rate == 0.0, f"seed {seed}"
