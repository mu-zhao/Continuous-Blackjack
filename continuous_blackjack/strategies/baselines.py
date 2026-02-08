from __future__ import annotations

import numpy as np

from continuous_blackjack.core.strategy import (
    BaseStrategy,
    DynamicCriticalValueStrategy,
    StaticCriticalValueStrategy,
)


class UniformStrategy(BaseStrategy):
    """Random critical-value baseline used for sanity checks."""

    def __init__(self, mode: str = "uninformed") -> None:
        if mode not in {"uninformed", "informed"}:
            raise ValueError("mode must be 'uninformed' or 'informed'")
        super().__init__(name=f"Uniform({mode})")
        self.mode = mode
        self.rng = np.random.default_rng()

    def calibrate(
        self,
        position,
        order,
        current_scores,
        current_round_hands,
        last_round,
    ) -> None:
        table_max = max(current_scores) if current_scores else 0.0
        if self.mode == "uninformed":
            self._critical_value = float(self.rng.random())
        else:
            self._critical_value = float(self.rng.uniform(table_max, 1.0))


class ZeroIntelligenceStrategy(StaticCriticalValueStrategy):
    """Fixed critical value at 0.5."""

    def __init__(self) -> None:
        super().__init__(lambda n: [0.5] * n)


class NaiveStrategy(DynamicCriticalValueStrategy):
    """Always uses current maximum on the table."""

    def __init__(self) -> None:
        super().__init__(lambda n: [0.0] * n)
