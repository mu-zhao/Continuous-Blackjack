"""PlatForm for continous blackjack"""

import time

import numpy as np
import pandas as pd


class PlatForm:
    def __init__(self, player_strategies):
        self._num_player = len(player_strategies)
        self._cum_reward = np.zeros(self._num_player)
        # Cumulative rewards breakdown by position
        self._reward_breakdown = np.zeros((self._num_player, self._num_player))
        # Bet score of the current round
        self._players = [player.initialize_stategy(player_id, self.num_player)
                         for player_id, player in enumerate(player_strategies)]

    def run(self, num_thousand_rounds=1000):
        start_time = time.time()
        self._cur_res = np.zeros(self._num_player)
        self._last_round = None
        # Sectional view of cumulative reward.
        self._sec_reward_history = np.zeros((num_thousand_rounds,
                                             self._num_player))
        for k in range(1, num_thousand_rounds + 1):
            for _ in range(1000):
                # Initialize the bet result of current round
                self._cur_res[:] = 0
                # The sum of n independent standard uniform variable
                # has prob 1/n! to be less than 1, for our purpose,
                # n=16 is good enough.
                self._pre_hands = np.cumsum(np.random.sample((
                    self.num_player, 16)), axis=1)
                self._post_hands = []
                # Reshuffle of players
                self.order = np.random.permutation(self._num_player)
                # Play the round
                winner_position = self.play_single_round()
                # Info of last round, read-only
                self._last_round = (winner_position, tuple(self.order),
                                    tuple(self._cur_res),
                                    tuple(self._post_hands))
                # Update reward history
                self._cum_reward[self.order[winner_position]] += 1
                self._reward_breakdown[self.order[winner_position],
                                       winner_position] += 1
            cur_time = time.time()
            print('current thousand rounds take %s seconds' % (
                cur_time - start_time))
            start_time = cur_time
            print('cumulative result up to %s thousand rounds:' % k,
                  self._cum_reward)

    def play_single_round(self):
        for position in range(self._num_player):
            strategy = self._players[self.order[position]]
            # Copy current info in case it get altered in the strategy algo.
            strategy.calibrate(position, self.order, tuple(self._cur_res),
                               tuple(self._post_hands), self._last_round)
            # Update current hand result.
            critical_index = self.deal(strategy, position)
            if self._pre_hands[position][critical_index] < 1:
                self._cur_res = self._pre_hands[position][critical_index]
            post_hand = tuple(self._pre_hands[position][:critical_index])
            self._post_hands.append(post_hand)

    def deal(self, strategy, position):
        # If the strategy use critical value as decision criteria.
        if strategy.fast_deal:
            critical_index = np.argmax(self._pre_hands[position] >
                                       strategy.critical_value)
        # Here card is the cumulative hand.
        for critical_index, card in enumerate(self._pre_hands[position]):
            if card > 1 or strategy.decision(card):
                break
        return critical_index

    def post_game_analysis(self):
        s = np.sum(self._reward_breakdown, axis=1)
        print('total reward:\n ', s / s[0])
        print('breakdown by position:\n ', self.rewards/self.rewards[0])
        print('percentege breakdown: \n',self.rewards/s)
        return pd.DataFrame(self.record),self.rewards
