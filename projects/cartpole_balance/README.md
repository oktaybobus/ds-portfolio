# CartPole Balance

A DQN, and the two-line rule that beats it.

| | |
|---|---|
| Task | Control (reinforcement learning) |
| Environment | `CartPole-v1`, solved at 475 of a maximum 500 |
| **Tuned DQN** | **500.0, solves 100%** [0.981, 1.000] |
| Two-line heuristic | 490.1, solves 93.5% |
| DQN with the notebook's settings | 197.1, solves **0%** |
| Random | 22.0 |
| Source | `day13-AOB-ReinforcementLearning.ipynb` |

```bash
uv run python projects/cartpole_balance/train.py
uv run python projects/cartpole_balance/train.py --skip-dqn   # baselines only, no torch
```

Needs `uv sync --extra deeprl` for the DQN; `--extra rl` is enough for the
baselines. Both DQNs train in under 16 seconds on a laptop CPU.

## The notebook trained it and stopped

```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)
model.learn(total_timesteps=50_000)
model.save("cartpole_dqn")
```

That is the last cell. The model is saved and the notebook ends, so the obvious
question - is it any good? - is never asked.

It is not. Over 200 episodes it averages **197 of a possible 500** and solves
the environment **0%** of the time. `verbose=1` prints a training reward that
climbs, which looks like success and is not the same measurement.

## Two lines of physics do better

```python
action = int(pole_angle + pole_angular_velocity > 0)
```

Push the cart the way the pole is falling. No network, no replay buffer, no
training, and two of the four observations ignored entirely.

| Agent | Mean return | Solves it | 95% interval |
|---|---|---|---|
| random | 22.0 | 0% | [0.000, 0.019] |
| **heuristic** | **490.1** | **93.5%** | [0.892, 0.962] |
| DQN, notebook settings | 197.1 | 0% | [0.000, 0.019] |
| DQN, tuned | 500.0 | 100% | [0.981, 1.000] |

The heuristic scores **2.5x** the notebook's DQN. It is the baseline that
should have been run first, because it costs nothing and it decides whether the
50,000 timesteps bought anything.

## The algorithm was never the problem

The same DQN, the same 50,000 timesteps, with the tuned configuration from RL
Baselines3 Zoo, scores a perfect 500 on every one of 200 episodes. What changed:

| | Notebook | Tuned |
|---|---|---|
| `train_freq` / `gradient_steps` | 4 / 1 (defaults) | 256 / 128 |
| `net_arch` | [64, 64] (default) | [256, 256] |
| `learning_rate` | 1e-3 | 2.3e-3 |
| `target_update_interval` | 10,000 (default) | 10 |
| `exploration_fraction` | 0.1 (default) | 0.16 |
| Training time | 7.9s | 15.1s |

The defaults take one gradient step every four environment steps. The tuned
configuration takes 128 every 256 - roughly eight times as much learning from
the same experience, which on CartPole is what the gap comes down to. Eight extra seconds separate an
agent that fails outright from one that is perfect.

So the notebook's DQN did not fail because deep RL is finicky, or because
50,000 steps is too few. It failed on hyper-parameters - and the only way to
find that out is to measure, which is the step that was missing.

## What the interval is for

Every row above carries one. At 200 episodes the heuristic's solve rate is
93.5% [0.892, 0.962], so it is genuinely below the tuned DQN rather than
unluckily so - the intervals do not overlap. Against the notebook's DQN no
interval is needed, but reporting one costs nothing and removes the question.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
