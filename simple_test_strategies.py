import numpy as np
from common_utils import BaseStrategy, CritValueStrategy, DynamicCVStartegy


class UniformStrategy(BaseStrategy):
    """this strategy is for testing
    """
    def __init__(self, random_type):
        self.algo_id = 'random strategy ' + random_type
        self.type = random_type
        self._fast_deal = True
        self._critical_value = 0

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        if self.type == 'uninformed':
            self._critical_value = np.random.sample()
        elif self.type == 'informed':
            self._critical_value = np.random.uniform(max(cur_res), 1)


class ZeroIntelStrategy(CritValueStrategy):
    """This is zero interlligence strategy, take 0.5 as critival value"""
    def __init__(self):
        super().__init__(lambda n: [0.5] * n)


class NaiveStrategy(DynamicCVStartegy):
    """ This strategy takes the previous max as critical value
    """
    def __init__(self):
        super().__init__(lambda n: [0] * n)
