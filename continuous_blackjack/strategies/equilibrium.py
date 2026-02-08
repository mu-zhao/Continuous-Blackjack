from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from continuous_blackjack.core.strategy import BaseStrategy, DynamicCriticalValueStrategy
from continuous_blackjack.core.types import RoundRecord

NUMERIC_TOLERANCE = 1e-4


def expected_stopping_payoff(x: float, num_nash: int, num_uniform: int) -> float:
    """Expected win probability when stopping immediately at score x."""
    return (1 - (1 - x) * np.exp(x)) ** num_nash * (2 - np.e - x + np.exp(x)) ** num_uniform


def stopping_advantage(x: float, num_nash: int, num_uniform: int) -> float:
    """Stopping payoff minus expected payoff from one more hit."""
    continuation = quad(expected_stopping_payoff, x, 1, args=(num_nash, num_uniform))[0]
    return expected_stopping_payoff(x, num_nash, num_uniform) - continuation


def critical_value_solution(num_players: int, tolerance: float = NUMERIC_TOLERANCE) -> np.ndarray:
    """Solve critical values by binary search over stopping advantage."""
    if num_players < 1:
        raise ValueError("num_players must be positive")

    result = np.zeros((num_players, num_players), dtype=float)
    for players_after in range(1, num_players):
        for num_uniform in range(players_after + 1):
            lo, hi = 0.0, 1.0
            while hi - lo > tolerance:
                mid = (hi + lo) / 2
                if stopping_advantage(mid, players_after - num_uniform, num_uniform) > 0:
                    hi = mid
                else:
                    lo = mid
            result[num_players - players_after - 1, num_uniform] = lo
    return result


class NashEquilibriumStrategy(DynamicCriticalValueStrategy):
    """Nash-equilibrium critical value strategy with optional scaling."""

    def __init__(self, scaling_factor: float = 1.0, tolerance: float = NUMERIC_TOLERANCE) -> None:
        if scaling_factor <= 0:
            raise ValueError("scaling_factor must be positive")
        self.scaling_factor = scaling_factor
        self.tolerance = tolerance
        super().__init__(lambda n: self.scaling_factor * critical_value_solution(n, self.tolerance)[:, 0])
        if scaling_factor != 1.0:
            self.set_name(f"NashEquilibrium(scale={scaling_factor:g})")


class AdaptiveNashEquilibriumStrategy(BaseStrategy):
    """Profiles opponents as rational vs uninformed and adapts critical values."""

    def __init__(self, confidence_rounds: int = 1000) -> None:
        super().__init__()
        if confidence_rounds <= 0:
            raise ValueError("confidence_rounds must be positive")
        self.confidence_rounds = confidence_rounds
        self._critical_table: np.ndarray | None = None
        self._profiles: np.ndarray | None = None
        self._processed_round: RoundRecord | None = None

    def set_parameters(self) -> None:
        self._critical_table = critical_value_solution(self.num_players)
        self._profiles = np.zeros(self.num_players, dtype=int)
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
            self._process_history(last_round)
            self._processed_round = last_round

        assert self._critical_table is not None
        assert self._profiles is not None

        uninformed_count = sum(
            self._profiles[player_id] < self.confidence_rounds
            for player_id in order[position + 1 :]
        )
        table_max = max(current_scores) if current_scores else 0.0
        self._critical_value = max(
            table_max,
            float(self._critical_table[position, uninformed_count]),
        )

    def _process_history(self, last_round: RoundRecord) -> None:
        assert self._profiles is not None

        max_result = 0.0
        for player_id, result, post_hand in zip(
            last_round.order,
            last_round.scores,
            last_round.post_hands,
        ):
            if player_id == self.player_id:
                max_result = max(max_result, result)
                continue

            if result > max_result:
                self._profiles[player_id] += 1
                max_result = result
            elif post_hand and post_hand[-1] < 1.0:
                # Stopping while below prior max is treated as irrational.
                self._profiles[player_id] = 0
