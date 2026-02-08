from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuous_blackjack.core.strategy import BaseStrategy
from continuous_blackjack.core.types import RoundRecord

FLOAT_EPS = 1e-12


@dataclass
class OpponentProfile:
    bust_prob: np.ndarray
    weight: np.ndarray


class StatisticalStrategy(BaseStrategy):
    """Estimates opponents' bust probabilities conditioned on table state."""

    def __init__(
        self,
        *,
        discretization_size: int = 1000,
        extrapolation_limit: float = 0.2,
        extrapolation_decay: float = 0.8,
        extrapolation_range: int = 5,
    ) -> None:
        super().__init__()
        if discretization_size <= 1:
            raise ValueError("discretization_size must be greater than 1")
        if not 0 <= extrapolation_limit <= 1:
            raise ValueError("extrapolation_limit must be in [0, 1]")
        if not 0 < extrapolation_decay <= 1:
            raise ValueError("extrapolation_decay must be in (0, 1]")
        if extrapolation_range < 0:
            raise ValueError("extrapolation_range must be non-negative")

        self.discretization_size = discretization_size
        self.extrapolation_limit = extrapolation_limit
        self.extrapolation_decay = extrapolation_decay
        self.extrapolation_range = extrapolation_range
        self.exp_values = np.exp(np.arange(discretization_size) / discretization_size)
        self.rng = np.random.default_rng()

        self.profiles: list[OpponentProfile | None] = []
        self._processed_round: RoundRecord | None = None

    def set_parameters(self) -> None:
        self.profiles = [None] * self.num_players
        for player_id in range(self.num_players):
            if player_id == self.player_id:
                continue
            self.profiles[player_id] = OpponentProfile(
                bust_prob=np.zeros((self.num_players, self.discretization_size), dtype=float),
                weight=np.ones((self.num_players, self.discretization_size), dtype=float),
            )
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

        table_max = max(current_scores) if current_scores else 0.0
        self._critical_value = table_max
        if position + 1 >= self.num_players:
            return

        bust_likelihood = np.ones(self.discretization_size, dtype=float)
        for later_position in range(position + 1, self.num_players):
            profile = self._profile_for(order[later_position])
            bust_likelihood *= profile.bust_prob[later_position]

        crit_idx = int(np.argmax(np.cumsum(bust_likelihood[::-1])[::-1] * self.exp_values))
        self._critical_value = max(
            table_max,
            float((crit_idx + self.rng.random()) / self.discretization_size),
        )

    def _process_history(self, last_round: RoundRecord) -> None:
        table_max = 0.0
        for position, (player_id, result) in enumerate(zip(last_round.order, last_round.scores)):
            if player_id == self.player_id:
                table_max = max(table_max, result)
                continue

            profile = self._profile_for(player_id)
            t_idx = min(int(table_max * self.discretization_size), self.discretization_size - 1)
            is_bust = float(result < FLOAT_EPS)

            if table_max < self.extrapolation_limit:
                left = max(0, t_idx - self.extrapolation_range)
                right = min(self.discretization_size, t_idx + self.extrapolation_range + 1)
                idx = np.arange(left, right)
                factors = self.extrapolation_decay ** np.abs(idx - t_idx)

                profile.weight[position, left:right] += factors
                profile.bust_prob[position, left:right] += (
                    (is_bust - profile.bust_prob[position, left:right])
                    * factors
                    / profile.weight[position, left:right]
                )
            else:
                profile.weight[position, t_idx] += 1.0
                profile.bust_prob[position, t_idx] += (
                    is_bust - profile.bust_prob[position, t_idx]
                ) / profile.weight[position, t_idx]

            table_max = max(table_max, result)

    def _profile_for(self, player_id: int) -> OpponentProfile:
        profile = self.profiles[player_id]
        if profile is None:
            raise RuntimeError("player profile is not initialized")
        return profile
