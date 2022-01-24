import random
from collections import namedtuple, deque
import torch
import numpy as np
from common_utils import BaseStrategy

rng = np.random.default_rng()
PastEps = namedtuple('PastExperience', ['state', 'reward'])
Experience = namedtuple('Experience', ('state', 'action', 'reward'))


def generate_state_feature(x, pos, order):
    num_player = len(order)
    log_n, exp_x = np.log(num_player), np.exp(x)
    feature = torch.zeros(8 + num_player + num_player**2)
    feature[:8] = torch.tensor([
        x, exp_x, x * exp_x, num_player,
        1 / num_player, log_n / num_player,
        log_n**2 / num_player, log_n**2 / num_player**2])
    for i, player in enumerate(order):
        if i < pos:
            feature[8 + player] = 1
        feature[8 + num_player + num_player * i + player] = 1
    return feature.double()


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, item):
        """Save an Experience"""
        self.memory.append(item)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NNStrategy(BaseStrategy):
    def __init__(self, nn_model, feature_func, output_size, *args):
        self.strategy = nn_model(output_size, *args)
        self.feature_func = feature_func
        self.size = output_size
        super().__init__()

    def set_parameter(self):
        super().set_parameter()
        self.strategy.initialize(self._num_player)

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        cur_state = self.feature_func(max(cur_res), position, order)
        if last_round:
            winner_pos, last_order = last_round[:2]
            last_reward = 1 if self._player_id == last_order[winner_pos] else 0
            self.strategy.process(last_reward)
        self._critical_value = self.strategy.action(cur_state) / self.size
