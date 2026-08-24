"""Tests for the reinforcement-learning helpers.

Most of this file needs no environment at all: the statistics that turn "it
ran" into "it scored X, give or take Y" are ordinary arithmetic and are tested
as such. The ones that step an environment are marked ``needs_rl``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dsjourney import rl


def test_wilson_interval_matches_the_published_value() -> None:
    low, high = rl.wilson_interval(74, 100)
    assert low == pytest.approx(0.6452, abs=0.002)
    assert high == pytest.approx(0.8177, abs=0.002)


def test_wilson_interval_stays_inside_zero_and_one() -> None:
    """Where the normal approximation goes out of range, this must not."""
    low, high = rl.wilson_interval(0, 20)
    assert low == 0.0
    assert high < 0.2
    low, high = rl.wilson_interval(20, 20)
    # Exactly 1 in algebra; the clamp keeps floating point from exceeding it.
    assert high <= 1.0
    assert high == pytest.approx(1.0)
    assert low > 0.8


def test_wilson_interval_narrows_with_more_trials() -> None:
    narrow = rl.wilson_interval(740, 1000)
    wide = rl.wilson_interval(74, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_on_no_trials() -> None:
    assert rl.wilson_interval(0, 0) == (0.0, 0.0)


def test_episodes_for_precision_is_the_textbook_count() -> None:
    """At p=0.74, two percentage points needs about 1,850 episodes."""
    assert rl.episodes_for_precision(0.74) == pytest.approx(1848, abs=5)
    assert rl.episodes_for_precision(0.74, half_width=0.05) < 400


def test_episodes_for_precision_grows_as_precision_tightens() -> None:
    assert rl.episodes_for_precision(0.5, half_width=0.01) > rl.episodes_for_precision(
        0.5, half_width=0.05
    )


def test_inverse_erf_rejects_impossible_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        rl.wilson_interval(1, 10, confidence=1.5)


def test_evaluation_result_reports_the_interval_with_the_number() -> None:
    result = rl.EvaluationResult(
        episodes=100,
        mean_return=0.74,
        return_sd=0.44,
        success_rate=0.74,
        ci_low=0.65,
        ci_high=0.82,
        mean_length=12.0,
    )
    assert "0.740" in result.summary()
    assert "[0.650, 0.820]" in result.summary()
    assert result.ci_half_width == pytest.approx(0.085)
    assert result.as_metrics("q_")["q_success_rate"] == 0.74


def test_geometric_schedule_hits_the_floor_regardless_of_budget() -> None:
    """The defect: 0.995 per episode reaches the floor after ~920 episodes.

    The notebook budgeted 20,000, so 95% of the run happens at 1% exploration
    on whatever the table held at episode 920 - which is often nothing.
    """
    config = rl.QLearningConfig(episodes=20_000, schedule="geometric")
    assert config.epsilon_at(920) == pytest.approx(config.epsilon_min, abs=1e-6)
    at_floor = sum(
        1 for episode in range(config.episodes) if config.epsilon_at(episode) <= config.epsilon_min
    )
    assert at_floor / config.episodes > 0.95


def test_linear_schedule_spends_the_whole_budget() -> None:
    config = rl.QLearningConfig(episodes=20_000, schedule="linear", exploration_fraction=0.5)
    assert config.epsilon_at(0) == pytest.approx(1.0)
    assert config.epsilon_at(5_000) == pytest.approx(0.505, abs=0.01)
    assert config.epsilon_at(10_000) == pytest.approx(config.epsilon_min, abs=1e-6)


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown schedule"):
        rl.QLearningConfig(schedule="cosine").epsilon_at(0)


def test_undecided_states_finds_untrained_rows() -> None:
    table = np.array([[0.0, 0.0], [0.3, 0.1], [0.0, 0.0]])
    assert rl.undecided_states(table).tolist() == [0, 2]
    assert rl.undecided_states(table, ignore=np.array([2])).tolist() == [0]


def test_greedy_policy_picks_the_best_action() -> None:
    table = np.array([[0.1, 0.9], [0.7, 0.2]])
    policy = rl.greedy_policy(table)
    assert policy(0) == 1
    assert policy(1) == 0


def test_greedy_policy_on_an_all_zero_row_answers_zero() -> None:
    """Not a bug in argmax - the reason undecided_states exists."""
    assert rl.greedy_policy(np.zeros((2, 4)))(0) == 0


def test_install_hint_names_both_extras() -> None:
    assert "--extra rl" in rl.INSTALL_HINT
    assert "--extra deeprl" in rl.INSTALL_HINT


def test_available_models_lists_the_configured_algorithms() -> None:
    assert set(rl.available_models()) == {"q_learning", "value_iteration", "dqn"}


@pytest.mark.needs_rl
def test_make_env_refuses_to_open_a_window() -> None:
    """A rendered toy-text env needs pygame and blocks without a display."""
    env = rl.make_env("FrozenLake-v1", render_mode="human")
    assert env.render_mode is None
    env.close()


@pytest.mark.needs_rl
def test_terminal_states_of_frozenlake() -> None:
    """Four holes and the goal, on the 4x4 map."""
    assert rl.terminal_states("FrozenLake-v1").tolist() == [5, 7, 11, 12, 15]


@pytest.mark.needs_rl
def test_value_iteration_solves_the_deterministic_lake() -> None:
    table, values, sweeps = rl.value_iteration("FrozenLake-v1", env_kwargs={"is_slippery": False})
    assert sweeps < 20
    assert values[0] == pytest.approx(0.951, abs=0.01)
    result = rl.evaluate_policy(
        "FrozenLake-v1",
        rl.greedy_policy(table),
        episodes=200,
        seed=3,
        env_kwargs={"is_slippery": False},
    )
    assert result.success_rate == 1.0


@pytest.mark.needs_rl
def test_an_exact_solution_leaves_no_state_undecided() -> None:
    table, _, _ = rl.value_iteration("FrozenLake-v1")
    terminals = rl.terminal_states("FrozenLake-v1")
    assert rl.undecided_states(table, ignore=terminals).size == 0


@pytest.mark.needs_rl
def test_random_policy_is_the_floor() -> None:
    result = rl.evaluate_policy(
        "FrozenLake-v1", rl.random_policy("FrozenLake-v1"), episodes=300, seed=5
    )
    assert result.success_rate < 0.1


@pytest.mark.needs_rl
@pytest.mark.slow
def test_q_learning_reaches_the_optimum_on_the_linear_schedule() -> None:
    config = rl.QLearningConfig(episodes=4_000, schedule="linear", seed=0)
    run = rl.train_q_learning("FrozenLake-v1", config, env_kwargs={"is_slippery": False})
    result = rl.evaluate_policy(
        "FrozenLake-v1",
        rl.greedy_policy(run.q_table),
        episodes=200,
        seed=3,
        env_kwargs={"is_slippery": False},
    )
    assert result.success_rate == 1.0
    assert len(run.curve) == 4_000
    assert "rolling_return" in run.curve.columns


@pytest.mark.needs_rl
@pytest.mark.slow
def test_compare_seeds_returns_one_row_per_seed() -> None:
    config = rl.QLearningConfig(episodes=1_500, schedule="linear")
    table = rl.compare_seeds(
        "FrozenLake-v1", config, seeds=3, eval_episodes=100, env_kwargs={"is_slippery": False}
    )
    assert len(table) == 3
    assert table["seed"].tolist() == [0.0, 1.0, 2.0]
    assert {"success_rate", "ci_low", "ci_high", "undecided_states"} <= set(table.columns)
