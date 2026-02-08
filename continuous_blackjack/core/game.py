from __future__ import annotations

import copy
import time
from typing import Iterable

import numpy as np

from continuous_blackjack.core.strategy import BaseStrategy
from continuous_blackjack.core.types import RoundRecord


class ContinuousBlackjackGame:
    """Simulation engine for continuous blackjack."""

    def __init__(
        self,
        player_strategies: Iterable[BaseStrategy],
        *,
        cards_per_player: int = 16,
        rng: np.random.Generator | None = None,
    ) -> None:
        strategies = list(player_strategies)
        if len(strategies) < 2:
            raise ValueError("ContinuousBlackjackGame requires at least 2 players")
        if cards_per_player <= 0:
            raise ValueError("cards_per_player must be positive")

        # A shared strategy instance across multiple seats causes state leakage.
        # Clone to enforce one strategy object per player seat.
        if len({id(strategy) for strategy in strategies}) < len(strategies):
            strategies = [copy.deepcopy(strategy) for strategy in strategies]

        self.num_players = len(strategies)
        self.cards_per_player = cards_per_player
        self.rng = rng or np.random.default_rng()

        self.cumulative_reward = np.zeros(self.num_players, dtype=float)
        self.reward_breakdown = np.zeros((self.num_players, self.num_players), dtype=float)
        self.players = [
            strategy.initialize(player_id, self.num_players)
            for player_id, strategy in enumerate(strategies)
        ]
        self.strategy_names = [strategy.name for strategy in self.players]
        self.player_labels = [
            f"p{player_id}:{strategy.name}"
            for player_id, strategy in enumerate(self.players)
        ]

        self._current_scores = np.zeros(self.num_players, dtype=float)
        self._last_round: RoundRecord | None = None
        self._section_reward_history = np.zeros((0, self.num_players), dtype=float)
        self._rounds_per_block = 0

    def run(
        self,
        *,
        num_blocks: int = 1000,
        rounds_per_block: int = 10_000,
        log: bool = True,
    ) -> np.ndarray:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if rounds_per_block <= 0:
            raise ValueError("rounds_per_block must be positive")

        self._last_round = None
        self._section_reward_history = np.zeros((num_blocks, self.num_players), dtype=float)
        self._rounds_per_block = rounds_per_block
        block_start = time.time()

        for block in range(num_blocks):
            for _ in range(rounds_per_block):
                winner_id = self._play_single_round()
                self.cumulative_reward[winner_id] += 1.0
                self._section_reward_history[block, winner_id] += 1.0

            if log:
                elapsed = time.time() - block_start
                block_start = time.time()
                block_share = self._section_reward_history[block] / rounds_per_block
                named_share = ", ".join(
                    f"{label}={value:.3f}"
                    for label, value in zip(self.player_labels, block_share)
                )
                print(f"block={block + 1} time={elapsed:.3f}s reward_share=[{named_share}]")

        return self._section_reward_history.copy()

    def _play_single_round(self) -> int:
        self._current_scores.fill(0.0)
        pre_hands = np.cumsum(
            self.rng.random((self.num_players, self.cards_per_player)),
            axis=1,
        )
        post_hands: list[tuple[float, ...]] = []
        order = self.rng.permutation(self.num_players)
        order_tuple = tuple(int(i) for i in order)

        for position, player_id in enumerate(order):
            strategy = self.players[int(player_id)]
            strategy.calibrate(
                position,
                order_tuple,
                tuple(float(v) for v in self._current_scores),
                tuple(post_hands),
                self._last_round,
            )

            stop_index = self._deal(strategy, pre_hands[position])
            stop_total = float(pre_hands[position, stop_index])
            if stop_total < 1.0:
                self._current_scores[position] = stop_total
            post_hands.append(tuple(float(v) for v in pre_hands[position, : stop_index + 1]))

        winner_position = int(np.argmax(self._current_scores))
        winner_id = order_tuple[winner_position]
        self.reward_breakdown[winner_id, winner_position] += 1.0
        self._last_round = RoundRecord(
            winner_position=winner_position,
            order=order_tuple,
            scores=tuple(float(v) for v in self._current_scores),
            post_hands=tuple(post_hands),
        )
        return winner_id

    def _deal(self, strategy: BaseStrategy, pre_hand: np.ndarray) -> int:
        if strategy.fast_deal:
            idx = int(np.searchsorted(pre_hand, strategy.critical_value, side="right"))
            return min(idx, len(pre_hand) - 1)

        for idx, card in enumerate(pre_hand):
            if card > 1.0 or strategy.decision(float(card)):
                return idx
        return len(pre_hand) - 1

    def summary(self) -> dict[str, object]:
        reward_by_strategy: dict[str, float] = {}
        for strategy_name, reward in zip(self.strategy_names, self.cumulative_reward):
            reward_by_strategy[strategy_name] = reward_by_strategy.get(strategy_name, 0.0) + float(reward)

        total_rounds = float(np.sum(self.cumulative_reward))
        if total_rounds > 0:
            player_win_rate = self.cumulative_reward / total_rounds
            position_win_rate = self.reward_breakdown / total_rounds
        else:
            player_win_rate = np.zeros_like(self.cumulative_reward)
            position_win_rate = np.zeros_like(self.reward_breakdown)

        player_total_wins = np.sum(self.reward_breakdown, axis=1)
        position_given_win = np.divide(
            self.reward_breakdown,
            player_total_wins[:, None],
            out=np.zeros_like(self.reward_breakdown),
            where=player_total_wins[:, None] > 0,
        )
        if self._rounds_per_block > 0:
            block_reward_share = self._section_reward_history / self._rounds_per_block
        else:
            block_reward_share = np.zeros_like(self._section_reward_history)

        player_summary = [
            {
                "player_id": player_id,
                "label": label,
                "strategy": strategy_name,
                "wins": float(wins),
                "win_rate": float(win_rate),
            }
            for player_id, (label, strategy_name, wins, win_rate) in enumerate(
                zip(
                    self.player_labels,
                    self.strategy_names,
                    self.cumulative_reward,
                    player_win_rate,
                )
            )
        ]
        player_summary_sorted = tuple(
            sorted(player_summary, key=lambda item: item["wins"], reverse=True)
        )

        seats_by_strategy: dict[str, int] = {}
        for strategy_name in self.strategy_names:
            seats_by_strategy[strategy_name] = seats_by_strategy.get(strategy_name, 0) + 1

        strategy_summary = [
            {
                "strategy": strategy_name,
                "seats": seats_by_strategy[strategy_name],
                "wins": reward,
                "win_rate": (reward / total_rounds) if total_rounds > 0 else 0.0,
            }
            for strategy_name, reward in reward_by_strategy.items()
        ]
        strategy_summary_sorted = tuple(
            sorted(strategy_summary, key=lambda item: item["wins"], reverse=True)
        )

        return {
            "total_rounds": total_rounds,
            "cumulative_reward": self.cumulative_reward.copy(),
            "player_win_rate": player_win_rate.copy(),
            "reward_breakdown": self.reward_breakdown.copy(),
            "position_win_rate": position_win_rate.copy(),
            "position_given_win": position_given_win.copy(),
            "section_reward_history": self._section_reward_history.copy(),
            "block_reward_share": block_reward_share.copy(),
            "strategy_names": tuple(self.strategy_names),
            "player_labels": tuple(self.player_labels),
            "reward_by_strategy": reward_by_strategy,
            "player_summary": player_summary_sorted,
            "strategy_summary": strategy_summary_sorted,
        }

    def post_game_analysis(self):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for post_game_analysis()") from exc

        summary = self.summary()
        reward_sum = np.sum(self.reward_breakdown, axis=1)
        baseline = reward_sum[0] if reward_sum[0] > 0 else 1.0
        row_baseline = self.reward_breakdown[0]
        row_baseline_safe = np.where(row_baseline == 0, 1.0, row_baseline)

        print("player labels:\n", np.array(self.player_labels, dtype=object))
        print("total reward ratio:\n", reward_sum / baseline)
        print("breakdown by position ratio:\n", self.reward_breakdown / row_baseline_safe)
        percentage_breakdown = np.divide(
            self.reward_breakdown,
            reward_sum[:, None],
            out=np.zeros_like(self.reward_breakdown),
            where=reward_sum[:, None] > 0,
        )
        print("percentage breakdown:\n", percentage_breakdown)
        print("total rounds:", int(summary["total_rounds"]))
        print("top players:", summary["player_summary"][:3])

        breakdown_df = pd.DataFrame(
            self.reward_breakdown,
            index=pd.Index(self.player_labels, name="Player"),
            columns=[f"Position {position + 1}" for position in range(self.num_players)],
        )
        reward_series = pd.Series(self.cumulative_reward, index=self.player_labels, name="reward")
        strategy_series = pd.Series(self.strategy_names, index=self.player_labels, name="strategy")
        strategy_totals = reward_series.groupby(strategy_series).sum().sort_values(ascending=False)
        print("reward by strategy:\n", strategy_totals)
        return breakdown_df
