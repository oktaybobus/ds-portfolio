# CartPole Balance

A DQN, and the two-line rule that beats it.

| | |
|---|---|
| Task | Control (reinforcement learning) |
| Environment | `CartPole-v1`, solved at 475 of a maximum 500 |
| Two-line heuristic | 490.1, solves 93.5% |
| **Tuned DQN**, median of 6 seeds | **~500** - but one seed in six lands at 105 |
| DQN with the notebook's settings | 126-202 across 6 seeds, solves **0%** on all of them |
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

It is not. Over 200 episodes it averages **197 of a possible 500**, where 475
is what CartPole calls solved, and not one of those 200 episodes cleared the
bar. Trained again on six different seeds it lands between 124 and 202, so that
is a property of the configuration rather than of the run - though the odd
episode does clear 475 by luck, and CI saw one in fifty. `verbose=1` prints a
training reward that climbs, which looks like success and is not the same
measurement.

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
| DQN, notebook settings (seed 0) | 197.1 | 0% | [0.000, 0.019] |

The heuristic scores **2.5x** the notebook's DQN, and it beats every one of the
six seeds that configuration was trained on. It is the baseline that should
have been run first, because it costs nothing and it decides whether the 50,000
timesteps bought anything.

## The algorithm was never the problem - but it is not reliable either

The same DQN, the same 50,000 timesteps, with the tuned configuration from RL
Baselines3 Zoo, reaches 500 on most seeds. What changed:

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

So the notebook's DQN did not fail because 50,000 steps is too few. It failed
on hyper-parameters - and the only way to find that out is to measure, which is
the step that was missing.

### This project made the same mistake, and CI caught it

The first version of this README said the tuned DQN "scores a perfect 500 on
every one of 200 episodes". That was one seed on one machine. Trained six
times:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Tuned DQN | 500.0 | 500.0 | 500.0 | 489.6 | **104.9** | 499.1 |
| Notebook DQN | 202.4 | 143.3 | 125.7 | 126.0 | 157.3 | 198.0 |

One seed in six collapses to 105. And on the Linux CI runner, seed 0 - the one
that scores 500 here - produced **18.9**, below random, because DQN on this
budget is sensitive to the platform's floating-point details as well as to the
seed.

The notebook's configuration is *consistently* bad: 126 to 202, never solving,
on every seed and both platforms. That claim survives. "The tuned one scores
500" did not, and it was the same defect this project was written to document -
[a stochastic result reported from one run](../../docs/tr/tekrar-eden-hatalar.md).

`train.py --seeds N` now trains each configuration N times and reports the
**median** seed with the range beside it. Never the best one.

How many seeds is the same question as how many episodes, and it has the same
kind of answer. Against a failure that happens on one seed in six, three seeds
miss it 58% of the time and five miss it 40% of the time. The default is five
because the runs cost fifteen seconds each; the table above needed six. A sweep
that finds nothing is evidence in proportion to its size, and no more.

## Two sources of noise, and they are different

An interval over episodes and a spread over seeds answer different questions,
and this project needs both:

- **Episodes** measure one trained agent. At 200 episodes the heuristic's solve
  rate is 93.5% [0.892, 0.962]. More episodes narrow that and nothing else.
- **Seeds** measure the training procedure. No number of evaluation episodes
  would have revealed that one tuned run in six collapses, because each run was
  evaluated perfectly well - the variation is in what training produced.

The first version of this project had the first and not the second, which is
exactly enough to be confidently wrong.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
