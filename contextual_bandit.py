from collections import namedtuple
from itertools import combinations
import numpy as np
from common_utils import BaseStrategy

BanditHistory = namedtuple('BanditHistory', ['reward', 'count'])
_SOFT = 1e-6

class ContextualBandits(BaseStrategy):
    """Base contexutal bandit strategy"""
    def __init__(self, size=100, resource_limit=3, init_reward=2):
        super().__init__()
        self.size = size
        self.resource_limit = resource_limit
        self._init_reward = init_reward
        self.rounds_done = 0
        self.last_state_action = None

    def set_parameter(self):
        self.dict = {}
        self.num_bandits = self.code()
        # row major for performance
        self.bandits = BanditHistory(
            np.zeros((self.num_bandits, self.size)) + self._init_reward,
            np.ones((self.num_bandits, self.size), dtype=int))

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        self.rounds_done += 1
        if last_round and self.last_state_action:
            self.process_history(last_round)
        self._critical_value = max(cur_res)
        if position < self._num_player - 1:
            if position < self.resource_limit:
                bandit_idx = self.dict[(tuple(sorted(order[:position])), 0)]
            elif position >= self._num_player - self.resource_limit:
                bandit_idx = self.dict[
                    (tuple(sorted(order[position + 1:])), 1)]
            else:
                bandit_idx = self.dict[position]
            low_action_index = int(self._critical_value * self.size)
            action = self.action(bandit_idx, low_action_index)
            self._critical_value = (action + np.random.sample()) / self.size
            self.last_state_action = (bandit_idx, action)
        else:
            self.last_state_action = None

    def process_history(self, last_round):
        winner_position, order = last_round[:2]
        self.pre_reward = order[winner_position] == self._player_id
        # Update bandits' history
        self.bandits.count[self.last_state_action] += 1
        self.bandits.reward[self.last_state_action] += (
            self.pre_reward - self.bandits.reward[self.last_state_action]
            ) / self.bandits.count[self.last_state_action]

    def code(self):
        k = 0
        for j in range(self.resource_limit,
                       self._num_player - self.resource_limit):
            self.dict[j] = k
            k += 1
        L = [i for i in range(self._num_player) if i != self._player_id]
        for j in range(self.resource_limit):
            for c in combinations(L, j):
                for d in range(2):
                    self.dict[(c, d)] = k
                    k += 1
        return k

    def action(self, bandit_index, low_action_index):
        raise NotImplementedError


class Greedy(ContextualBandits):
    def __init__(self, xp, xp_decay, xp_decay_rounds, *args) -> None:
        self.xp = xp
        self._xp_decay = xp_decay
        self._xp_decay_rounds = xp_decay_rounds
        super().__init__(*args)

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        if self.rounds_done % self._xp_decay_rounds == 0:
            self.xp *= self._xp_decay
        super().calibrate(position, order, cur_res, cur_round_hands,
                          last_round)

    def action(self, bandit_index, low_action_index):
        if np.random.sample() < self.xp:
            return np.random.randint(low_action_index, self.size)
        return low_action_index + np.argmax(
            self.bandits.reward[bandit_index, low_action_index:])


class UCBStrategy(ContextualBandits):
    def __init__(self, confidence_level, *args):
        self.c = confidence_level
        super().__init__(*args)

    def action(self, bandit_index, low_action_index):
        return low_action_index + np.argmax(
            self.bandits.reward[bandit_index, low_action_index:] +
            self.c * np.sqrt(np.log(self.rounds_done) / self.bandits.count[
                bandit_index, low_action_index:]))


class PolicyGradient(ContextualBandits):
    def __init__(self, baseline, lr, *args):
        self.baseline = baseline
        self.lr = lr
        super().__init__(*args)

    def set_parameter(self):
        super().set_parameter()
        self.H = np.zeros((self.num_bandits, self.size)) + self.baseline

    def action(self, bandit_index, low_action_index):
        action_porb = np.exp(self.H[bandit_index, low_action_index:]) + _SOFT
        action_porb /= sum(action_porb)
        return low_action_index + np.random.choice(
            self.size - low_action_index, 1, p=action_porb)

    def process_history(self, last_round):
        super().process_history(last_round)
        bandit_idx = self.last_state_action[0]
        self.H[bandit_idx] -= self.lr * (
            self.pre_reward - self.bandits.reward[self.last_state_action] *
            np.exp(self.H[bandit_idx]))
        self.H[self.last_state_action] += self.lr * (
            self.pre_reward - self.bandits.reward[self.last_state_action])
        if self.H[bandit_idx].max() > 5 * self.baseline:
            self.H[bandit_idx] -= 4 * self.baseline
