# Continuous Blackjack

Refactored simulation framework for continuous blackjack strategy research.

## Game Rules

1. Players are randomly re-ordered every round.
2. On a turn, a player repeatedly draws `U(0, 1)` and accumulates the sum.
3. A player stops according to its strategy.
4. A score is the stopping sum if it is `< 1`, otherwise `0`.
5. Highest score wins the round (ties go to the earliest max in current implementation).

## Project Layout

```
continuous_blackjack/
  core/          # engine, round record, strategy base classes
  strategies/    # equilibrium, adaptive, statistical, bandits
  rl/            # optional torch-based DQN and actor-critic
experiments/
  run_experiment.py
tests/
```

## Install

Python requirement: `3.14+`

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[dev,analysis,rl]"
```

Or with requirements files:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-analysis.txt
python -m pip install -r requirements-rl.txt
python -m pip install -r requirements-dev.txt
```

## Run Example Simulation

```bash
python experiments/run_experiment.py --set-index 1 --blocks 20 --rounds-per-block 1000 --log
```

Actor-Critic experiment:

```bash
python experiments/run_experiment.py --actor-critic --blocks 20 --rounds-per-block 1000 --log
# or:
python experiments/run_actor_critic_experiment.py --blocks 20 --rounds-per-block 1000 --log
```

Save/load RL model parameters (DQN / Actor-Critic):

```bash
python experiments/run_experiment.py --actor-critic --blocks 20 --rounds-per-block 1000 --save-model-dir checkpoints/ac
python experiments/run_experiment.py --actor-critic --blocks 20 --rounds-per-block 1000 --load-model-dir checkpoints/ac --save-model-dir checkpoints/ac
```

Notebook:

- `/Users/muzhao/Documents/Workspace/Python/Continuous-Blackjack/experiments/continuous_blackjack_experiment.ipynb`
- Configure `load_model_dir` / `save_model_dir` in the parameter cell to resume RL models.

## Run Tests

```bash
python -m pytest
```

## Notes

- No backward-compatibility layer is provided in this refactor.
- RL modules require `torch`; core and non-RL strategies do not.
