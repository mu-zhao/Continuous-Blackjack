from __future__ import annotations

import numpy as np

from continuous_blackjack.strategies import critical_value_solution


def test_critical_value_solution_shape_and_range():
    table = critical_value_solution(5)
    assert table.shape == (5, 5)
    assert np.all(table >= 0)
    assert np.all(table <= 1)


def test_critical_value_solution_empty_last_row():
    # Last row corresponds to no players after current player.
    table = critical_value_solution(4)
    assert np.allclose(table[-1], 0)
