# FrozenLake Control

Tabular Q-learning, judged against the exact optimum instead of against nothing.

| | |
|---|---|
| Task | Control (reinforcement learning) |
| Environment | `FrozenLake-v1`, 4x4, slippery |
| **Success rate** | **0.726 [0.706, 0.745]** over 2,000 episodes |
| Exact optimum (value iteration) | 0.726 - the agent reaches it |
| Random policy | 0.013 |
| Seeds that learn nothing | 0 of 12 (the notebook's schedule: 2 of 12) |
| Source | `day13-AOB-ReinforcementLearning.ipynb` |

```bash
uv run python projects/frozenlake_control/train.py
uv run python projects/frozenlake_control/train.py --schedule geometric --no-slippery
```

Needs `uv sync --extra rl`. No JVM, no GPU, no download - the environment is
sixteen squares.

## The notebook's evaluation was one episode

```python
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    time.sleep(0.5)
    action = np.argmax(Q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    ...
print("total reward: ", total_reward)
```

That is the whole check on 20,000 episodes of training. On the slippery lake it
prints `1.0` or `0.0`, and the optimal policy - the best that can possibly
exist - prints `0.0` about a quarter of the time, because the ice moves you
sideways and sometimes there is nothing to be done about it.

An evaluation that reports total failure for a perfect agent in one run out of
four is not a weak evaluation. It is a coin flip with a `print` on the end.

`evaluate_policy` never returns a bare number:

| Episodes | Result |
|---|---|
| 1 | 0.0 or 1.0 |
| 100 | 0.72 [0.63, 0.80] |
| 2,000 | 0.726 [0.706, 0.745] |

`episodes_for_precision(0.726)` says 1,913 episodes to pin the rate to two
percentage points. One resolves nothing at all.

## Run it again and you get a different answer

The bigger problem is upstream. Q-learning is stochastic, the notebook trained
it once, and the run it happened to get is the run it reported. Twelve seeds of
the notebook's own configuration:

| Setting | Schedule | Mean | Median | Seeds that learned nothing |
|---|---|---|---|---|
| **deterministic** (the notebook's) | `epsilon *= 0.995` | 0.417 | 0.000 | **7 of 12** |
| deterministic | linear over half the run | **1.000** | 1.000 | 0 of 12 |
| slippery | `epsilon *= 0.995` | 0.605 | 0.725 | 2 of 12 |
| slippery | linear over half the run | **0.727** | 0.726 | 0 of 12 |

On the notebook's own settings, seven runs in twelve produce an agent that
never once reaches the goal. Its results are `[0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]` -
not a spread around a mean, but two outcomes with nothing in between. The
notebook drew a 1.

## Why: the schedule ignores the budget

`epsilon *= 0.995` after each episode reaches the 0.01 floor at episode 920,
whatever was budgeted. Of 20,000 episodes, 19,080 run at one percent
exploration on whatever the table held at episode 920.

If the agent has not stumbled onto the goal by then, the table is still zeros,
`np.argmax` of a zero row returns action 0 - LEFT - and from the start square
LEFT does nothing. The agent presses against the wall for 100 steps, gets
truncated, and repeats that for 19,000 episodes. Reward is never seen, so
nothing is ever learned, and no error is raised: the run simply ends.

Spreading the same decay linearly over half the budget fixes it outright: 12
seeds of 12 reach the optimum on both lakes. That is the only change.

## `argmax` over zeros is not a decision

A state the agent never visited keeps a row of zeros, and `np.argmax` answers
`0` - a real action, chosen by tie-breaking. `undecided_states` counts them,
excluding the five terminal squares where a zero row is correct.

In the failed runs above every non-terminal state is undecided, so the failure
is visible in the table itself and not only in the score. The exact solution
from value iteration leaves none, which is the test.

## Deterministic FrozenLake is not a reinforcement-learning problem

The notebook used `is_slippery=False`, where every action does exactly what it
says. That is a shortest-path puzzle on sixteen squares; value iteration solves
it in **7 sweeps**, and the answer is a fixed sequence of moves. The slippery
version needs 420 sweeps and tops out at 0.726, because a third of the time the
ice takes you somewhere else.

Both are worth running - `--no-slippery` switches - but only one of them is
about learning under uncertainty, and the difference is one keyword argument
that the notebook never revisited.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
