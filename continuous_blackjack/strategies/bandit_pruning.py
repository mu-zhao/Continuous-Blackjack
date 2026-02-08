from __future__ import annotations

import math

import numpy as np

from continuous_blackjack.strategies.contextual_bandits import ContextualBanditStrategy


class BanditPruningStrategy(ContextualBanditStrategy):
    """Contextual bandit with periodic action-set pruning."""

    def __init__(
        self,
        *,
        algorithm: str = "epsilon_greedy",
        prune_interval: int = 20_000,
        keep_fraction: float = 0.75,
        min_count_for_pruning: int = 25,
        exploration_rate: float = 0.15,
        exploration_decay: float = 0.9,
        confidence_level: float = 2.0,
        baseline: float = 3.0,
        learning_rate: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if algorithm not in {"epsilon_greedy", "ucb", "policy_gradient"}:
            raise ValueError(
                "algorithm must be one of: epsilon_greedy, ucb, policy_gradient"
            )
        if prune_interval <= 0:
            raise ValueError("prune_interval must be positive")
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if min_count_for_pruning <= 0:
            raise ValueError("min_count_for_pruning must be positive")

        self.algorithm = algorithm
        self.prune_interval = prune_interval
        self.keep_fraction = keep_fraction
        self.min_count_for_pruning = min_count_for_pruning

        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.confidence_level = confidence_level
        self.baseline = baseline
        self.learning_rate = learning_rate

        self.active_actions: np.ndarray | None = None
        self.action_steps: np.ndarray | None = None
        self.preferences: np.ndarray | None = None
        self.set_name(f"BanditPruning({algorithm})")

    def set_parameters(self) -> None:
        super().set_parameters()
        self.active_actions = np.ones((self.num_bandits, self.num_actions), dtype=bool)
        self.action_steps = np.zeros(self.num_bandits, dtype=int)
        self.preferences = None
        if self.algorithm == "policy_gradient":
            self.preferences = np.full((self.num_bandits, self.num_actions), self.baseline)

    def action(self, bandit_index: int, low_action_index: int) -> int:
        assert self.bandits is not None
        assert self.active_actions is not None
        assert self.action_steps is not None

        candidates = np.flatnonzero(self.active_actions[bandit_index, low_action_index:])
        if len(candidates) == 0:
            return low_action_index
        candidates = candidates + low_action_index

        if self.algorithm == "epsilon_greedy":
            if self.rng.random() < self.exploration_rate:
                choice = int(self.rng.choice(candidates))
            else:
                rewards = self.bandits.reward[bandit_index, candidates]
                choice = int(candidates[int(np.argmax(rewards))])
            self.exploration_rate *= self.exploration_decay
        elif self.algorithm == "ucb":
            rounds = max(self.rounds_done, 2)
            rewards = self.bandits.reward[bandit_index, candidates]
            counts = self.bandits.count[bandit_index, candidates]
            bonus = self.confidence_level * np.sqrt(np.log(rounds) / counts)
            choice = int(candidates[int(np.argmax(rewards + bonus))])
        else:
            assert self.preferences is not None
            logits = self.preferences[bandit_index, candidates]
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs /= probs.sum()
            choice = int(self.rng.choice(candidates, p=probs))

        self.action_steps[bandit_index] += 1
        if self.action_steps[bandit_index] % self.prune_interval == 0:
            self._prune_bandit(bandit_index)
        return choice

    def process_history(self, last_round) -> None:
        super().process_history(last_round)
        if self.algorithm != "policy_gradient":
            return
        if self.last_state_action is None or self.preferences is None or self.bandits is None:
            return

        bandit_idx, action_idx = self.last_state_action
        baseline_reward = self.bandits.reward[bandit_idx, action_idx]
        logits = self.preferences[bandit_idx] - np.max(self.preferences[bandit_idx])
        probs = np.exp(logits)
        probs /= probs.sum()
        advantage = self.last_reward - baseline_reward

        self.preferences[bandit_idx] -= self.learning_rate * advantage * probs
        self.preferences[bandit_idx, action_idx] += self.learning_rate * advantage

    def _prune_bandit(self, bandit_index: int) -> None:
        assert self.bandits is not None
        assert self.active_actions is not None

        active_idx = np.flatnonzero(self.active_actions[bandit_index])
        if len(active_idx) <= 2:
            return

        counts = self.bandits.count[bandit_index, active_idx]
        eligible = active_idx[counts >= self.min_count_for_pruning]
        if len(eligible) <= 2:
            return

        rewards = self.bandits.reward[bandit_index, eligible]
        keep_n = max(2, math.ceil(self.keep_fraction * len(eligible)))
        keep_rank = np.argsort(rewards)[-keep_n:]
        keep_actions = set(int(a) for a in eligible[keep_rank])

        for action_idx in eligible:
            if int(action_idx) not in keep_actions:
                self.active_actions[bandit_index, int(action_idx)] = False
