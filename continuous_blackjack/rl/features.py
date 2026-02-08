from __future__ import annotations

import numpy as np
import torch


def generate_state_feature(table_max: float, position: int, order) -> torch.Tensor:
    num_players = len(order)
    log_n = np.log(num_players)
    exp_x = np.exp(table_max)

    feature = torch.zeros(8 + num_players + num_players**2, dtype=torch.double)
    feature[:8] = torch.tensor(
        [
            table_max,
            exp_x,
            table_max * exp_x,
            num_players,
            1 / num_players,
            log_n / num_players,
            log_n**2 / num_players,
            log_n**2 / num_players**2,
        ],
        dtype=torch.double,
    )
    for turn_position, player_id in enumerate(order):
        if turn_position < position:
            feature[8 + player_id] = 1.0
        feature[8 + num_players + num_players * turn_position + player_id] = 1.0

    return feature
