
from collections import namedtuple
import numpy as np
from common_utils import BaseStrategy

Profile = namedtuple('Profile', ['L', 'weight'])
_FLOAT_ERR = 1e-12


class StatisticalStrategy(BaseStrategy):
    """ This strategy makes the assumption that all players are
        rational. For each player this strategy maintain a profile
        to estimate $L$ as opposed to $K$, where
        L(t) is the prob of the said player goes bust when the previous
        max valid result is t.
        The estimate of $L(t)$ for small $t$ is not as accurate
         as for that of larger $t$
        since there is less chance for the previous score to be small.
         We can fix that to some extent by extrapolation.
        however the estimate of $L$ for small $t$ does not matter
        too much as the critical value is likely to be found for larger
         $t$ where the estimates are relatively accurate.
    """
    def __init__(self, discretization_size=1000, extraplation_limt=0.2,
                 xp_decay_factor=0.8, extrapolation_range=5):
        self.size = discretization_size
        self.exp_val = np.exp(np.arange(self.size) / self.size)
        self.range = extrapolation_range
        self.xtp_limit = extraplation_limt
        # Extrapolation factor
        self.xtp_factor = xp_decay_factor**abs(
            np.arange(-self.range, self.range+1))

    def set_parameter(self):
        self.profiles = [0] * self._num_player
        for player_id in range(self._num_player):
            if player_id != self._player_id:
                self.profile[player_id] = Profile([
                    np.zeros((self._num_player, self.size)),
                    np.ones((self._num_player, self.size))])

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        self.process_history(last_round)
        self._critical_value = max(cur_res)
        if position + 1 < self._num_player:
            M = np.ones(self.size)
            for pos in range(position+1, self._num_player):
                M *= self.profiles[order[pos]].L[pos]

            optimal_crti_val_idx = np.argmax(
                np.cumsum(M[::-1])[::-1] * self.exp_val) + \
                np.random.random_sample()

            self._critical_value = max(self.critical_value,
                                       optimal_crti_val_idx / self.size)

    def process_history(self, last_round):
        if last_round:
            # only need last turn's results for positions and bet results
            # t is the max previous bet result
            t = 0
            for position, (player_id, bet_res) in enumerate(
                    zip(*last_round[1:3])):
                if player_id != self._player_id:
                    profile = self.profiles[player_id]
                    # index of the value t in the array(Profile.L)
                    t_index = int(t * self.size)
                    # If the bet_res is 0, then the player's hand went bust
                    is_bust = bet_res < _FLOAT_ERR
                    # If t < 0.2, it's a small value, we use extrapolation to
                    # get more data by exponential decay.
                    if t < self.xtp_limit:
                        l_idx = max(0, t_index - self.range)
                        r_idx = t_index + self.range + 1

                        profile.weight[position, l_idx: r_idx] += (
                            self.xtp_factor[l_idx - r_idx:])

                        profile.L[position, l_idx: r_idx] += (
                            is_bust*self.xtp_factor[l_idx - r_idx:] -
                            profile.L[position, l_idx: r_idx]
                        ) / profile.weight[position, l_idx: r_idx]
                    else:
                        profile.weight[position, t_index] += 1
                        profile.L[position, t_index] += (
                            is_bust - profile.L[position, t_index]
                            ) / profile.weight[position, t_index]
                t = max(t, bet_res)
