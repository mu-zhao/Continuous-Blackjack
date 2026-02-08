from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Callable, Sequence


class BaseStrategy(ABC):
    """Shared strategy protocol for the game engine."""

    def __init__(self, *, fast_deal: bool = True, name: str | None = None) -> None:
        self._fast_deal = fast_deal
        self._critical_value = 0.0
        self._name = name or self.__class__.__name__
        self.player_id = -1
        self.num_players = 0

    def initialize(self, player_id: int, num_players: int) -> BaseStrategy:
        self.player_id = player_id
        self.num_players = num_players
        self.set_parameters()
        return self

    def set_parameters(self) -> None:
        """Reset strategy state for a new game."""

    @abstractmethod
    def calibrate(
        self,
        position: int,
        order: Sequence[int],
        current_scores: Sequence[float],
        current_round_hands: Sequence[Sequence[float]],
        last_round,
    ) -> None:
        """Update critical value before a player's turn."""

    def decision(self, card_total: float) -> bool:
        """Used by non-fast strategies. Return True when stopping."""
        return True

    @property
    def critical_value(self) -> float:
        return float(self._critical_value)

    @property
    def name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        if not name:
            raise ValueError("name must be a non-empty string")
        self._name = name

    @property
    def fast_deal(self) -> bool:
        return bool(self._fast_deal)


class RandomizedStrategy(BaseStrategy):
    """Periodically switches between a pool of strategy factories."""

    def __init__(
        self,
        strategy_factories: Sequence[Callable[[], BaseStrategy]],
        switch_interval: int,
    ) -> None:
        super().__init__()
        if switch_interval <= 0:
            raise ValueError("switch_interval must be positive")
        if not strategy_factories:
            raise ValueError("strategy_factories must not be empty")
        self._factories = strategy_factories
        self.switch_interval = switch_interval
        self._rounds_done = 0
        self._current_strategy: BaseStrategy | None = None

    def set_parameters(self) -> None:
        self._rounds_done = 0
        self._swap_strategy()

    def _swap_strategy(self) -> None:
        strategy = random.choice(self._factories)()
        self._current_strategy = strategy.initialize(self.player_id, self.num_players)

    def calibrate(
        self,
        position: int,
        order: Sequence[int],
        current_scores: Sequence[float],
        current_round_hands: Sequence[Sequence[float]],
        last_round,
    ) -> None:
        if self._current_strategy is None:
            self._swap_strategy()
        if self._rounds_done >= self.switch_interval:
            self._swap_strategy()
            self._rounds_done = 0

        assert self._current_strategy is not None
        self._current_strategy.calibrate(
            position,
            order,
            current_scores,
            current_round_hands,
            last_round,
        )
        self._critical_value = self._current_strategy.critical_value
        self._fast_deal = self._current_strategy.fast_deal
        self._rounds_done += 1

    def decision(self, card_total: float) -> bool:
        if self._current_strategy is None:
            return True
        return self._current_strategy.decision(card_total)


class StaticCriticalValueStrategy(BaseStrategy):
    """Critical values are fixed by position."""

    def __init__(self, critical_value_fn: Callable[[int], Sequence[float]]) -> None:
        super().__init__()
        self._critical_value_fn = critical_value_fn
        self._critical_values: tuple[float, ...] = ()

    def set_parameters(self) -> None:
        values = tuple(float(v) for v in self._critical_value_fn(self.num_players))
        if len(values) != self.num_players:
            raise ValueError(
                "critical_value_fn must return one critical value per player"
            )
        self._critical_values = values

    def calibrate(
        self,
        position: int,
        order: Sequence[int],
        current_scores: Sequence[float],
        current_round_hands: Sequence[Sequence[float]],
        last_round,
    ) -> None:
        self._critical_value = self._critical_values[position]


class DynamicCriticalValueStrategy(StaticCriticalValueStrategy):
    """Critical value is max(static value, current table maximum)."""

    def calibrate(
        self,
        position: int,
        order: Sequence[int],
        current_scores: Sequence[float],
        current_round_hands: Sequence[Sequence[float]],
        last_round,
    ) -> None:
        table_max = max(current_scores) if current_scores else 0.0
        self._critical_value = max(table_max, self._critical_values[position])
