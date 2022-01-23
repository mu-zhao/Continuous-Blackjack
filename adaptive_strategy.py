
import numpy as np
from common_utils import BaseStrategy
from nash_equilibrium_strategies import critical_value_solution

# TODO: need write more comments.


class ADProfile:
    """ this is the profile for AdaptiveStrategy, which are distribution
        families. To enhance the performance, I run two tests:
        1) lower bound test:
        lower bound means rationality assumption holds,
        i.e, the critical value will be greater than the max previous res
        2)point strategy test;
        point bound means the opponents' strategies are not randomized,
        that is, it's always a static cirtical value.
    """
    def __init__(self, player_id, num_players, const_G, init_w, gridsize=1000,
                 discount=0.99, pt_threshold=2500, lowbd_threshold=1500,
                 cooldown=10000):
        self.palyer_id = player_id
        self.pt_threshold = pt_threshold
        self.lowbd_threshold = lowbd_threshold
        self.grid = gridsize
        self.discount = discount
        self.const_G = const_G
        self.weight = [np.zeros(self.grid) + init_w
                       for _ in range(num_players)]

        self.const_dist = [np.zeros((self.grid, self.grid)) + 1 / self.grid
                     for _ in range(num_players)]
        self.init_sh = np.sum(self.const_G, axis=1) / self.grid
        self.sh = [np.copy(self.init_sh) for _ in range(num_players)]
        temp = np.zeros((self.grid, 2))
        temp[:, 1] = 1
        self.bounds = [np.copy(temp) for _ in range(num_players)]
        self.lowbd_established = [False] * num_players
        self.pt_established = [False] * num_players
        self.acceptable_error = 2 / self.grid
        self.lowbd_count = np.zeros(num_players, dtype=int)
        self.pt_count = np.zeros(num_players, dtype=int)
        self.cooldownturns = cooldown
        self.cooldown = 3 * cooldown

    def update(self, rank, t, a, b):
        self.cooldown -= 1
        if t <= b:
            self.lowbd_count[rank] += 1
            if not (self.lowbd_established[rank] or self.pt_established[rank]
                    ) and self.lowbd_count[rank] > self.lowbd_threshold:
                self.establish_lowbd(rank)
        elif self.lowbd_established[rank]:
            self.lowbd_breached(rank)
        A = int(self.grid * t)
        l, h = self.bounds[rank][A]
        if a < h + self.acceptable_error and b > l-self.acceptable_error:
            self.pt_count[rank] += 1
            self.bounds[rank][A] = max(l, a), min(b, h)
            if not self.pt_established[rank] and self.pt_count[
                    rank] > self.pt_threshold:
                self.establish_pt(rank)
        elif self.pt_established[rank]:
            self.pt_breached(rank)
        x = int(a*self.grid)
        y = min(int(self.grid * b) + 1, self.grid)
        if self.pt_established:
            self.pt_fit(rank, A)
        elif self.lowbd_established:
            self.lowbd_fit(rank, A, y)
        else:
            for k in range(-3, 4):
                A1, x1, y1 = A + k, x + k, y + k
                if 0 <= A1 < self.grid and 0 <= x1 and y1 < self.grid:
                    self.fit(rank, A1, x1, y1, 0.9**abs(k))
        if self.cooldown > 0:
            return True
        else:
            return False

    def lowbd_breached(self, rank):
        self.lowbd_established[rank] = False
        self.lowbd_count[rank] = 0
        self.sh[rank][:] = self.init_sh
        self.const_dist[rank][:] = 1 / self.grid
        self.cooldown = self.cooldownturns

    def establish_lowbd(self, rank):
        self.lowbd_established[rank] = True
        for A in range(self.grid):
            z = sum(self.const_dist[rank][A, A:])
            self.const_dist[rank][A, :A] = 0
            self.const_dist[rank][A][A:] /= z
            self.sh[rank][A] = sum(self.const_dist[rank][A, A:] *
                                   self.const_G[A, A:])

    def establish_pt(self, rank):
        self.pt_established[rank] = True
        for A in range(self.grid):
            self.pt_fit(rank, A)

    def pt_breached(self, rank):
        self.pt_established[rank] = False
        self.bounds[rank][:0] = 0
        self.bounds[rank][:1] = 1
        self.pt_count[rank] = 0
        self.sh[rank] = self.init_sh
        self.const_dist[rank][:] = 1 / self.grid
        self.cooldown = self.cooldownturns

    def pt_fit(self, rank, A):
        l, h = self.bounds[rank][A]
        x = max(0, int((l - self.acceptable_error) * self.grid))
        y = min(int((self.acceptable_error + h) * self.grid) + 1, self.grid)
        self.const_dist[rank][A][:] = 0
        self.const_dist[rank][A][x:y] = 1 / (y - x)
        self.sh[rank][A] = sum(self.init_sh[x:y]) * self.grid / (y - x)

    def lowbd_fit(self, rank, A, y):
        self.fit(rank, A, A, y)

    def fit(self, rank, A, x, y, extraplation_factor=1):
        Z = sum(self.const_dist[rank][A, x:y]) * self.discount
        w = np.zeros(self.grid) + self.weight[rank][A]
        delta = extraplation_factor * sum(
            self.const_dist[rank][A][x:y] * self.const_G[A][x:y]) / Z
        self.sh[rank][A] *= self.weight[rank][A]
        self.sh[rank][A] += delta
        self.weight[rank][A] += extraplation_factor / self.discount
        self.sh[rank][A] /= self.weight[rank][A]
        w[x:y] += extraplation_factor / Z
        w /= self.weight[rank][A]
        self.const_dist[rank][A] *= w

    def getsh(self, rank):
        return self.sh[rank]

    def getprofile(self, rank):
        return self.const_dist[rank]


class AdaptiveStrategy(BaseStrategy):
    """This was the model free strategy,  made before I
        realized that the conditions can be imposed on $L$
        instead of the strategy $K$.
    """
    def __init__(self, girdsize=100, discount=0.9, init_w=10,
                 pt_threshold=1500, lowbd_threshold=1000, cooldown=4000):
        super().__init__()
        self.grid = girdsize
        self.discount = discount
        self.init_w = init_w
        self.cooldown = cooldown
        self.pt_threshold = pt_threshold
        self.lowbd_threshold = lowbd_threshold
        temp = (np.arange(self.grid)+0.1) / self.grid
        self.exp_temp = np.exp(temp)
        res = 1 - (1 - temp) * self.exp_temp
        self.const_G = np.zeros((self.grid, self.grid))
        for t in range(self.grid):
            self.const_G[t] = res
            self.const_G[t, :t] = (
                t / self.grid - temp[:t]) * self.exp_temp[:t]

    def set_parameter(self):
        self.nashequilibrum = critical_value_solution(self._num_player)
        self.profiles = [
            ADProfile(i, self._num_player, self.const_G, self.init_w,
                      self.grid, self.discount, self.pt_threshold,
                      self.lowbd_threshold, self.cooldown)
            for i in range(self._num_player)]
        self.profiles[self._player_id] = None

    def calibrate(self, position, order, cur_res, cur_round_hands, last_round):
        self.iscooldown = False
        self.process_history(last_round)
        if position == self._num_player - 1:
            self._critical_value = max(cur_res)
        elif self.iscooldown:
            num_uninformed = 0
            for pos in range(position + 1, self._num_player):
                player = order[pos]
                if not (self.profiles[player].pt_established[pos] or
                        (self.profiles[player].lowbd_established[pos])):
                    num_uninformed += 1
            self._critical_value = max(
                max(cur_res), self.nashequilibrum[pos, num_uninformed])
        else:
            W = np.ones(self.grid)
            for i in range(position+1, self.num_players):
                W *= self.profiles[order[i]].getsh(i)
            self.crit_index = np.argmax(
                np.cumsum(W[::-1])[::-1] * self.exp_temp) + 0.5
            self._critical_value = max(self.crit_index / self.grid,
                                       max(cur_res))

    def process_history(self, last_round):
        if last_round:
            t = 0
            for pos, (player_id, res, hand) in enumerate(zip(*last_round[1:])):
                if player_id != self._player_id:
                    a = 0
                    if len(hand) > 1:
                        a = hand[-2]
                    b = min(1, hand[-1])
                    if self.profiles[player_id].update(pos, t, a, b):
                        self.iscooldown = True
                t = max(t, res)
