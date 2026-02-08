from __future__ import annotations

import numpy as np

from continuous_blackjack.core import ContinuousBlackjackGame
from continuous_blackjack.strategies import (
    NaiveStrategy,
    NashEquilibriumStrategy,
    UniformStrategy,
)


def test_game_smoke_run():
    game = ContinuousBlackjackGame(
        [
            NashEquilibriumStrategy(),
            NaiveStrategy(),
            UniformStrategy("uninformed"),
        ],
        cards_per_player=8,
        rng=np.random.default_rng(0),
    )
    block_history = game.run(num_blocks=1, rounds_per_block=200, log=False)
    assert block_history.shape == (1, 3)
    assert np.isclose(block_history.sum(), 200)

    summary = game.summary()
    assert summary["cumulative_reward"].shape == (3,)
    assert np.isclose(summary["cumulative_reward"].sum(), 200)
