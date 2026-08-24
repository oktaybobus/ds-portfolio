"""Tests for the FrozenLake control project.

The environment ships with gymnasium, so these assert facts about the real
4x4 lake rather than a fixture.
"""

from __future__ import annotations

import pytest

from dsjourney import rl
from projects.frozenlake_control import pipeline

STATES = 16
ACTIONS = 4
OPTIMAL_SLIPPERY = 0.726
OPTIMAL_DETERMINISTIC = 1.0


def test_the_map_is_the_published_one() -> None:
    tiles = pipeline.tile_kinds()
    assert len(tiles) == STATES
    assert "".join(tiles) == "SFFFFHFHFFFHHFFG"
    assert (tiles == "H").sum() == 4


def test_learning_config_comes_from_the_yaml() -> None:
    config = pipeline.learning_config()
    assert config.schedule == "linear"
    assert config.episodes == 20_000
    assert config.gamma == pytest.approx(0.99)


def test_learning_config_takes_overrides() -> None:
    assert pipeline.learning_config(schedule="geometric").schedule == "geometric"


def test_env_kwargs_default_to_the_configured_slipperiness() -> None:
    assert pipeline.env_kwargs() == {"is_slippery": True}
    assert pipeline.env_kwargs(slippery=False) == {"is_slippery": False}


def test_prepare_input_explains_the_right_entry_point() -> None:
    with pytest.raises(NotImplementedError, match=r"train\.py"):
        pipeline.prepare_input({})


@pytest.mark.needs_rl
def test_load_raw_returns_the_transition_table() -> None:
    frame = pipeline.load_raw(slippery=False)
    assert list(frame.columns) == [
        "state",
        "action",
        "probability",
        "next_state",
        "reward",
        "terminal",
    ]
    # Deterministic: exactly one outcome per (state, action).
    assert len(frame) == STATES * ACTIONS
    assert frame["probability"].eq(1.0).all()


@pytest.mark.needs_rl
def test_a_slippery_move_has_three_outcomes() -> None:
    frame = pipeline.load_raw(slippery=True)
    non_terminal = frame[~frame["state"].isin(rl.terminal_states(pipeline.ENV_ID))]
    counts = non_terminal.groupby(["state", "action"]).size()
    assert counts.eq(3).all()


@pytest.mark.needs_rl
@pytest.mark.slow
def test_value_iteration_gives_the_known_optimum() -> None:
    for slippery, expected in ((True, OPTIMAL_SLIPPERY), (False, OPTIMAL_DETERMINISTIC)):
        kwargs = pipeline.env_kwargs(slippery=slippery)
        table, _, _ = rl.value_iteration(pipeline.ENV_ID, env_kwargs=kwargs)
        result = rl.evaluate_policy(
            pipeline.ENV_ID,
            rl.greedy_policy(table),
            episodes=2000,
            seed=999,
            env_kwargs=kwargs,
        )
        assert result.success_rate == pytest.approx(expected, abs=0.03)


@pytest.mark.needs_rl
@pytest.mark.slow
def test_the_notebook_schedule_fails_on_seeds_the_linear_one_survives() -> None:
    """The finding this project exists for.

    On the notebook's own setting - deterministic lake, epsilon *= 0.995 - some
    seeds learn nothing whatsoever, because exploration reaches its floor after
    ~920 of the 20,000 budgeted episodes and the table is still empty. The
    notebook ran it once and reported the run it got.

    Four seeds and 6,000 episodes are enough to show the split without the
    twelve-seed sweep the README quotes.
    """
    kwargs = pipeline.env_kwargs(slippery=False)
    geometric = rl.compare_seeds(
        pipeline.ENV_ID,
        pipeline.learning_config(episodes=6_000, schedule="geometric"),
        seeds=4,
        eval_episodes=200,
        env_kwargs=kwargs,
    )
    linear = rl.compare_seeds(
        pipeline.ENV_ID,
        pipeline.learning_config(episodes=6_000, schedule="linear"),
        seeds=4,
        eval_episodes=200,
        env_kwargs=kwargs,
    )
    assert (linear["success_rate"] == 1.0).all(), "the linear schedule must solve every seed"
    assert (geometric["success_rate"] < 0.05).any(), "the notebook schedule must fail some seed"
    assert geometric["success_rate"].mean() < linear["success_rate"].mean()


@pytest.mark.needs_rl
@pytest.mark.slow
def test_a_failed_run_leaves_states_undecided() -> None:
    """A seed that learns nothing is visible in the table, not only in the score."""
    kwargs = pipeline.env_kwargs(slippery=False)
    table = rl.compare_seeds(
        pipeline.ENV_ID,
        pipeline.learning_config(episodes=6_000, schedule="geometric"),
        seeds=4,
        eval_episodes=200,
        env_kwargs=kwargs,
    )
    failures = table[table["success_rate"] < 0.05]
    assert len(failures) > 0
    # A failed run is diagnosable without the score: states it never learned.
    assert (failures["undecided_states"] > 0).all()
