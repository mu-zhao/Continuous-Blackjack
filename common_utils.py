from typing import final

import numpy as np


class BaseStrategy:
    def __init__(self):
        self._fast_deal = True
        self._critical_value = 0

    def initialize_stategy(self, player_id, num_players):
        self._player_id = player_id
        self._num_player = num_players
        self.set_parameter()
        return self

    def set_parameter(self):
        pass

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        raise NotImplementedError

    def decision(self, card):
        return True

    @final
    @property
    def critical_value(self):
        return self._critical_value

    @final
    @property
    def fast_deal(self):
        return self._fast_deal


class RandomizedStrategy(BaseStrategy):
    def __init__(self, alternative_strategies, switch_num):
        super().__init__()
        self._alt_strategies = alternative_strategies
        self.switch_num = switch_num

    def set_parameter(self):
        self.rounds_done = 0
        self._cur_strategy = np.random.choice(self._alt_strategies)()
        self._cur_strategy.set_parameter()

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        if self.rounds_done == self.switch_num:
            self.set_parameter()
        self._cur_strategy.calibrate(position, order, cur_res,
                                     cur_round_hands, last_round)
        self.rounds_done += 1

    def decision(self, card):
        return self._cur_strategy.decision(card)


class CritValueStrategy(BaseStrategy):
    """(Static) critical value strategy, critical value is predetermined."""
    def __init__(self, func_crit_value):
        self._func_crit_value = func_crit_value
        super().__init__()

    def set_parameter(self):
        self.static_crit_values = self._func_crit_value(self._num_player)
        assert len(self.static_crit_values) == self._num_player

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        self._critical_value = self.static_crit_values[position]


class DynamicCVStartegy(CritValueStrategy):
    """Dynamic critical strategy differs from static critical strategy
       in that its critical value will be
       the max of current results and predetermined critical value.
    """
    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        self._critical_value = max(max(cur_res),
                                   self.static_crit_values[position])
