from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from continuous_blackjack.core.strategy import BaseStrategy
from continuous_blackjack.core.types import RoundRecord


@dataclass
class BanditHistory:
    reward: np.ndarray
    count: np.ndarray


class ContextualBanditStrategy(BaseStrategy):
    """Base class for contextual bandit critical-value strategies."""

    def __init__(
        self,
        *,
        num_actions: int = 100,
        resource_limit: int = 3,
        initial_reward: float = 2.0,
    ) -> None:
        super().__init__()
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than 1")
        if resource_limit <= 0:
            raise ValueError("resource_limit must be positive")

        self.num_actions = num_actions
        self.resource_limit = resource_limit
        self.initial_reward = initial_reward
        self.rng = np.random.default_rng()

        self.rounds_done = 0
        self.bandit_index: dict[object, int] = {}
        self.num_bandits = 0
        self.bandits: BanditHistory | None = None
        self.last_state_action: tuple[int, int] | None = None
        self.last_reward = 0.0
        self._processed_round: RoundRecord | None = None

    def set_parameters(self) -> None:
        self.rounds_done = 0
        self.bandit_index = {}
        self.num_bandits = self._build_bandit_index()
        self.bandits = BanditHistory(
            reward=np.full((self.num_bandits, self.num_actions), self.initial_reward, dtype=float),
            count=np.ones((self.num_bandits, self.num_actions), dtype=int),
        )
        self.last_state_action = None
        self.last_reward = 0.0
        self._processed_round = None

    def calibrate(
        self,
        position: int,
        order,
        current_scores,
        current_round_hands,
        last_round: RoundRecord | None,
    ) -> None:
        if last_round is not None and last_round is not self._processed_round:
            if self.last_state_action is not None:
                self.process_history(last_round)
            self._processed_round = last_round

        self.rounds_done += 1
        table_max = max(current_scores) if current_scores else 0.0
        self._critical_value = table_max

        if position >= self.num_players - 1:
            self.last_state_action = None
            return

        if position < self.resource_limit:
            key = (tuple(sorted(order[:position])), 0)
        elif position >= self.num_players - self.resource_limit:
            key = (tuple(sorted(order[position + 1 :])), 1)
        else:
            key = position

        bandit_idx = self.bandit_index[key]
        low_action_idx = min(self.num_actions - 1, int(table_max * self.num_actions))
        action_idx = self.action(bandit_idx, low_action_idx)
        action_idx = int(np.clip(action_idx, low_action_idx, self.num_actions - 1))

        self._critical_value = float((action_idx + self.rng.random()) / self.num_actions)
        self.last_state_action = (bandit_idx, action_idx)

    def process_history(self, last_round: RoundRecord) -> None:
        if self.bandits is None or self.last_state_action is None:
            return

        won = 1.0 if last_round.winner_id == self.player_id else 0.0
        self.last_reward = won
        bandit_idx, action_idx = self.last_state_action

        self.bandits.count[bandit_idx, action_idx] += 1
        count = self.bandits.count[bandit_idx, action_idx]
        old_reward = self.bandits.reward[bandit_idx, action_idx]
        self.bandits.reward[bandit_idx, action_idx] = old_reward + (won - old_reward) / count

    def _build_bandit_index(self) -> int:
        index = 0
        for position in range(self.resource_limit, self.num_players - self.resource_limit):
            self.bandit_index[position] = index
            index += 1

        remaining_players = [i for i in range(self.num_players) if i != self.player_id]
        for prefix_size in range(self.resource_limit):
            for combo in combinations(remaining_players, prefix_size):
                for tail_flag in (0, 1):
                    self.bandit_index[(combo, tail_flag)] = index
                    index += 1
        return index

    def action(self, bandit_index: int, low_action_index: int) -> int:
        raise NotImplementedError


class EpsilonGreedyBanditStrategy(ContextualBanditStrategy):
    def __init__(
        self,
        *,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.99,
        exploration_decay_rounds: int = 10_000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be in [0, 1]")
        if not 0 < exploration_decay <= 1:
            raise ValueError("exploration_decay must be in (0, 1]")
        if exploration_decay_rounds <= 0:
            raise ValueError("exploration_decay_rounds must be positive")

        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_decay_rounds = exploration_decay_rounds
        self.set_name("BanditEpsilonGreedy")

    def calibrate(self, position, order, current_scores, current_round_hands, last_round) -> None:
        if self.rounds_done > 0 and self.rounds_done % self.exploration_decay_rounds == 0:
            self.exploration_rate *= self.exploration_decay
        super().calibrate(position, order, current_scores, current_round_hands, last_round)

    def action(self, bandit_index: int, low_action_index: int) -> int:
        assert self.bandits is not None
        if self.rng.random() < self.exploration_rate:
            return int(self.rng.integers(low_action_index, self.num_actions))
        offset = int(np.argmax(self.bandits.reward[bandit_index, low_action_index:]))
        return low_action_index + offset


class UCBBanditStrategy(ContextualBanditStrategy):
    def __init__(self, *, confidence_level: float = 3.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if confidence_level < 0:
            raise ValueError("confidence_level must be non-negative")
        self.confidence_level = confidence_level
        self.set_name(f"BanditUCB(c={confidence_level:g})")

    def action(self, bandit_index: int, low_action_index: int) -> int:
        assert self.bandits is not None
        rounds = max(self.rounds_done, 2)
        reward = self.bandits.reward[bandit_index, low_action_index:]
        count = self.bandits.count[bandit_index, low_action_index:]
        bonus = self.confidence_level * np.sqrt(np.log(rounds) / count)
        offset = int(np.argmax(reward + bonus))
        return low_action_index + offset


class PolicyGradientBanditStrategy(ContextualBanditStrategy):
    def __init__(self, *, baseline: float = 1.0, learning_rate: float = 0.01, **kwargs) -> None:
        super().__init__(**kwargs)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.baseline = baseline
        self.learning_rate = learning_rate
        self.preferences: np.ndarray | None = None
        self.set_name("BanditPolicyGradient")

    def set_parameters(self) -> None:
        super().set_parameters()
        self.preferences = np.full((self.num_bandits, self.num_actions), self.baseline, dtype=float)

    def action(self, bandit_index: int, low_action_index: int) -> int:
        assert self.preferences is not None
        logits = self.preferences[bandit_index, low_action_index:]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        offset = int(self.rng.choice(len(probs), p=probs))
        return low_action_index + offset

    def process_history(self, last_round: RoundRecord) -> None:
        super().process_history(last_round)
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
        self.preferences[bandit_idx] = np.clip(self.preferences[bandit_idx], -100.0, 100.0)
