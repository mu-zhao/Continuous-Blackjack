
import numpy as np
from scipy.integrate import quad

from common_utils import DynamicCVStartegy, BaseStrategy
_NUMERIC_ERR = 1e-4


def expected_stopping_payoff(x, num_nash, num_uniform):
    """ config: If the current player's hand is x, and
    There are num_player players after the current player,
    num_uniform of them play uniformed uniform strategy, i,e,
    the critical value is chosen unifromly from [0, 1]
    and the remaining num_nash players paly nash equilibrum.
    This function compute the exprected payoff if the current player
    stops here.
    If a later player beat the current player, they have a hand
    between x and 1.
    The probability of nash equilibrium player NOT beat current player
    is 1-(1-x)*np.exp(x),
    the prob of uninformed uniform player NOT beat current player is
    2-np.e-x+np.exp(x).

    """
    return (1 - (1 - x) * np.exp(x))**num_nash * (
        2 - np.e - x + np.exp(x))**num_uniform


# Integration
def stopping_advantage(A, num_nash, num_uniform):
    """ This function calculate the difference between the expected payoff
     of stopping right away and that of taking one more step,
      which is the integration.
    """
    return expected_stopping_payoff(A, num_nash, num_uniform) - quad(
        expected_stopping_payoff, A, 1, args=(num_nash, num_uniform))[0]


#
def critical_value_solution(n, err=_NUMERIC_ERR):
    "critical value is where the current player is neutral on stopping"
    res = np.zeros((n, n))
    for i in range(1, n):
        # j in the number of unifrom players.
        for j in range(i + 1):
            high, low = 1, 0
            while high - low > err:
                mid = (high + low) / 2
                if stopping_advantage(mid, i - j, j) > 0:
                    high = mid
                else:
                    low = mid
            res[n - i - 1, j] = low
    return res


class NashEquilibrium(DynamicCVStartegy):
    """ Nash equilibrium scaled by factor
    """
    def __init__(self, scaling_factor=1):
        super().__init__(
            lambda n: scaling_factor * critical_value_solution(n)[:, 0])


class AdaptiveNasheqilibrium(BaseStrategy):
    """ This strategy make the assumption that a player either Nash
     Equilibrium player or uniformend uniform player who pick a critical value
     uniformaly from $[0,1]$. More specifically, if rationality
    assumption is violated, the algorithm will mark the player
     as uninfromed uniform player.
    """
    def __init__(self, confidence_rounds=1000):
        # The number of consecutive rounds for a player not violating
        # rationaality to be marked as nash player.
        self.confidence_rounds = 1000
        self._func_crit_value = critical_value_solution
        super().__init__()

    def set_parameter(self):
        self.critical_value_table = self._func_crit_value(self._num_player)
        self.profiles = np.zeros(self._num_player, dtype=int)

    def calibrate(self, position, order, cur_res,
                  cur_round_hands, last_round):
        self.process_history(last_round)
        # If a player' profile value < confidence_rounds, then the
        # player is condidered as uninformed player.
        uninformed_player = 0
        for player_id in order[position + 1:]:
            uninformed_player += self.profiles[
                player_id] < self.confidence_rounds

        self._critical_value = max(max(cur_res), self.critical_value_table[
            position, uninformed_player])

    def process_history(self, last_round):
        max_res = 0
        if last_round:
            for player_id, res, post_hand in zip(*last_round[1:]):
                if max_res > 0:
                    if res > max_res:
                        self.profiles[player_id] += 1
                        max_res = res
                    # If a player stops before reaching the previous max, which
                    # violates rationality.
                    elif post_hand[-1] < 1:
                        self.profiles[player_id] = 0
