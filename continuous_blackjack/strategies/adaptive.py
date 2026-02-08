from __future__ import annotations

import numpy as np

from continuous_blackjack.core.strategy import BaseStrategy
from continuous_blackjack.core.types import RoundRecord
from continuous_blackjack.strategies.equilibrium import critical_value_solution


class AdaptiveProfile:
    """Opponent profile tracker used by AdaptiveStrategy."""

    def __init__(
        self,
        player_id: int,
        num_players: int,
        const_g: np.ndarray,
        initial_weight: float,
        *,
        grid_size: int = 1000,
        discount: float = 0.99,
        point_threshold: int = 2500,
        lower_bound_threshold: int = 1500,
        cooldown_rounds: int = 10_000,
    ) -> None:
        self.player_id = player_id
        self.point_threshold = point_threshold
        self.lower_bound_threshold = lower_bound_threshold
        self.grid_size = grid_size
        self.discount = discount
        self.const_g = const_g

        self.weight = [np.full(self.grid_size, initial_weight, dtype=float) for _ in range(num_players)]
        self.const_dist = [
            np.full((self.grid_size, self.grid_size), 1 / self.grid_size, dtype=float)
            for _ in range(num_players)
        ]
        self.init_sh = np.sum(self.const_g, axis=1) / self.grid_size
        self.sh = [self.init_sh.copy() for _ in range(num_players)]

        bounds_template = np.zeros((self.grid_size, 2), dtype=float)
        bounds_template[:, 1] = 1.0
        self.bounds = [bounds_template.copy() for _ in range(num_players)]

        self.lower_bound_established = [False] * num_players
        self.point_established = [False] * num_players
        self.acceptable_error = 2.0 / self.grid_size
        self.lower_bound_count = np.zeros(num_players, dtype=int)
        self.point_count = np.zeros(num_players, dtype=int)
        self.cooldown_rounds = cooldown_rounds
        self.cooldown = 3 * cooldown_rounds

    def update(self, rank: int, table_max: float, lower: float, upper: float) -> bool:
        self.cooldown -= 1

        if table_max <= upper:
            self.lower_bound_count[rank] += 1
            if (
                not self.lower_bound_established[rank]
                and not self.point_established[rank]
                and self.lower_bound_count[rank] > self.lower_bound_threshold
            ):
                self._establish_lower_bound(rank)
        elif self.lower_bound_established[rank]:
            self._lower_bound_breached(rank)

        table_idx = min(self.grid_size - 1, int(self.grid_size * table_max))
        low_bound, high_bound = self.bounds[rank][table_idx]
        if lower < high_bound + self.acceptable_error and upper > low_bound - self.acceptable_error:
            self.point_count[rank] += 1
            self.bounds[rank][table_idx] = max(low_bound, lower), min(upper, high_bound)
            if (
                not self.point_established[rank]
                and self.point_count[rank] > self.point_threshold
            ):
                self._establish_point(rank)
        elif self.point_established[rank]:
            self._point_breached(rank)

        x = min(self.grid_size - 1, int(lower * self.grid_size))
        y = min(int(self.grid_size * upper) + 1, self.grid_size)
        if self.point_established[rank]:
            self._fit_point(rank, table_idx)
        elif self.lower_bound_established[rank]:
            self._fit_lower_bound(rank, table_idx, y)
        else:
            for offset in range(-3, 4):
                table_idx_1 = table_idx + offset
                x_1 = x + offset
                y_1 = y + offset
                if 0 <= table_idx_1 < self.grid_size and 0 <= x_1 < y_1 <= self.grid_size:
                    self._fit(rank, table_idx_1, x_1, y_1, extrapolation_factor=0.9 ** abs(offset))

        return self.cooldown > 0

    def _lower_bound_breached(self, rank: int) -> None:
        self.lower_bound_established[rank] = False
        self.lower_bound_count[rank] = 0
        self.sh[rank][:] = self.init_sh
        self.const_dist[rank][:] = 1 / self.grid_size
        self.cooldown = self.cooldown_rounds

    def _establish_lower_bound(self, rank: int) -> None:
        self.lower_bound_established[rank] = True
        for table_idx in range(self.grid_size):
            mass = np.sum(self.const_dist[rank][table_idx, table_idx:])
            if mass <= 0:
                continue
            self.const_dist[rank][table_idx, :table_idx] = 0
            self.const_dist[rank][table_idx, table_idx:] /= mass
            self.sh[rank][table_idx] = np.sum(
                self.const_dist[rank][table_idx, table_idx:] * self.const_g[table_idx, table_idx:]
            )

    def _establish_point(self, rank: int) -> None:
        self.point_established[rank] = True
        for table_idx in range(self.grid_size):
            self._fit_point(rank, table_idx)

    def _point_breached(self, rank: int) -> None:
        self.point_established[rank] = False
        self.bounds[rank][:, 0] = 0.0
        self.bounds[rank][:, 1] = 1.0
        self.point_count[rank] = 0
        self.sh[rank] = self.init_sh.copy()
        self.const_dist[rank][:] = 1 / self.grid_size
        self.cooldown = self.cooldown_rounds

    def _fit_point(self, rank: int, table_idx: int) -> None:
        low_bound, high_bound = self.bounds[rank][table_idx]
        x = max(0, int((low_bound - self.acceptable_error) * self.grid_size))
        y = min(int((high_bound + self.acceptable_error) * self.grid_size) + 1, self.grid_size)
        y = max(y, x + 1)

        self.const_dist[rank][table_idx][:] = 0
        self.const_dist[rank][table_idx][x:y] = 1 / (y - x)
        self.sh[rank][table_idx] = np.sum(self.init_sh[x:y]) * self.grid_size / (y - x)

    def _fit_lower_bound(self, rank: int, table_idx: int, y: int) -> None:
        self._fit(rank, table_idx, table_idx, y)

    def _fit(
        self,
        rank: int,
        table_idx: int,
        x: int,
        y: int,
        *,
        extrapolation_factor: float = 1.0,
    ) -> None:
        z = np.sum(self.const_dist[rank][table_idx, x:y]) * self.discount
        if z <= 0:
            return

        row_weight = self.weight[rank][table_idx]
        weights = np.full(self.grid_size, row_weight, dtype=float)
        delta = (
            extrapolation_factor
            * np.sum(self.const_dist[rank][table_idx][x:y] * self.const_g[table_idx][x:y])
            / z
        )

        self.sh[rank][table_idx] *= row_weight
        self.sh[rank][table_idx] += delta
        self.weight[rank][table_idx] += extrapolation_factor / self.discount
        self.sh[rank][table_idx] /= self.weight[rank][table_idx]

        weights[x:y] += extrapolation_factor / z
        weights /= self.weight[rank][table_idx]
        self.const_dist[rank][table_idx] *= weights

    def get_sh(self, rank: int) -> np.ndarray:
        return self.sh[rank]


class AdaptiveStrategy(BaseStrategy):
    """Model-free adaptive approximation strategy."""

    def __init__(
        self,
        *,
        grid_size: int = 100,
        discount: float = 0.9,
        initial_weight: float = 10.0,
        point_threshold: int = 1500,
        lower_bound_threshold: int = 1000,
        cooldown_rounds: int = 4000,
    ) -> None:
        super().__init__()
        if grid_size <= 1:
            raise ValueError("grid_size must be greater than 1")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1]")

        self.grid_size = grid_size
        self.discount = discount
        self.initial_weight = initial_weight
        self.cooldown_rounds = cooldown_rounds
        self.point_threshold = point_threshold
        self.lower_bound_threshold = lower_bound_threshold

        temp = (np.arange(self.grid_size, dtype=float) + 0.1) / self.grid_size
        self.exp_temp = np.exp(temp)
        res = 1 - (1 - temp) * self.exp_temp

        self.const_g = np.zeros((self.grid_size, self.grid_size), dtype=float)
        for table_idx in range(self.grid_size):
            self.const_g[table_idx] = res
            self.const_g[table_idx, :table_idx] = (
                table_idx / self.grid_size - temp[:table_idx]
            ) * self.exp_temp[:table_idx]

        self.nash_equilibrium: np.ndarray | None = None
        self.profiles: list[AdaptiveProfile | None] = []
        self._processed_round: RoundRecord | None = None
        self._cooldown_active = False

    def set_parameters(self) -> None:
        self.nash_equilibrium = critical_value_solution(self.num_players)
        self.profiles = [
            AdaptiveProfile(
                player_id=player_id,
                num_players=self.num_players,
                const_g=self.const_g,
                initial_weight=self.initial_weight,
                grid_size=self.grid_size,
                discount=self.discount,
                point_threshold=self.point_threshold,
                lower_bound_threshold=self.lower_bound_threshold,
                cooldown_rounds=self.cooldown_rounds,
            )
            for player_id in range(self.num_players)
        ]
        self.profiles[self.player_id] = None
        self._processed_round = None
        self._cooldown_active = False

    def calibrate(
        self,
        position: int,
        order,
        current_scores,
        current_round_hands,
        last_round: RoundRecord | None,
    ) -> None:
        if last_round is not None and last_round is not self._processed_round:
            self._cooldown_active = self._process_history(last_round)
            self._processed_round = last_round

        assert self.nash_equilibrium is not None

        table_max = max(current_scores) if current_scores else 0.0
        if position == self.num_players - 1:
            self._critical_value = table_max
            return

        if self._cooldown_active:
            num_uninformed = 0
            for later_position in range(position + 1, self.num_players):
                profile = self._profile_for(order[later_position])
                if not (
                    profile.point_established[later_position]
                    or profile.lower_bound_established[later_position]
                ):
                    num_uninformed += 1
            self._critical_value = max(
                table_max,
                float(self.nash_equilibrium[position, num_uninformed]),
            )
            return

        w = np.ones(self.grid_size, dtype=float)
        for later_position in range(position + 1, self.num_players):
            w *= self._profile_for(order[later_position]).get_sh(later_position)

        crit_idx = np.argmax(np.cumsum(w[::-1])[::-1] * self.exp_temp) + 0.5
        self._critical_value = max(float(crit_idx / self.grid_size), table_max)

    def _process_history(self, last_round: RoundRecord) -> bool:
        cooldown_active = False
        table_max = 0.0
        for position, (player_id, result, hand) in enumerate(
            zip(last_round.order, last_round.scores, last_round.post_hands)
        ):
            if player_id != self.player_id:
                lower = hand[-2] if len(hand) > 1 else 0.0
                upper = min(1.0, hand[-1]) if hand else 0.0
                profile = self._profile_for(player_id)
                cooldown_active = profile.update(position, table_max, lower, upper) or cooldown_active
            table_max = max(table_max, result)
        return cooldown_active

    def _profile_for(self, player_id: int) -> AdaptiveProfile:
        profile = self.profiles[player_id]
        if profile is None:
            raise RuntimeError("player profile is not initialized")
        return profile
