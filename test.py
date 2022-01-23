#%%
from platform_utils import PlatForm as PF
from nash_equilibrium_strategies import NashEquilibrium, AdaptiveNasheqilibrium
from simple_test_strategies import (
    UniformStrategy, ZeroIntelStrategy, NaiveStrategy)
from adaptive_strategy import AdaptiveStrategy
from actor_acritic import ACStrategy
from QDN import DQNStrategy
from statistical_strategy import StatisticalStrategy
from contextual_bandit import Greedy, UCBStrategy, PolicyGradient
# TODO: rewrite bandit_pruning, do NOT use CBPruning.
from bandit_pruning import CBPruning


NE = NashEquilibrium()
ZS = ZeroIntelStrategy()
NS = NaiveStrategy()
ADNE = AdaptiveNasheqilibrium()
UnUF = UniformStrategy('uninformed')
UF = UniformStrategy('imformed')
ADS = AdaptiveStrategy()
SS = StatisticalStrategy()
UCB = UCBStrategy(confidence_level=3)
GD = Greedy(xp=0.1, xp_decay=0.99, xp_decay_rounds=10000)
PG = PolicyGradient(baseline=1, lr=0.01)
DQN = DQNStrategy(output_size=1024, lr=0.001)
ACS = ACStrategy(output_size=1024, alpha=0.01, beta=0.01)


Strategy_sets = [
    [NE, UnUF, UF, SS],
    [UnUF, UF, NE, GD],
    [ZS, NE, ADNE],
    [ZS, NS, UCB],
    [UnUF, UF, NS, PG],
    [UnUF, ZS, NE, DQN],
    [UnUF, ZS, NE, ACS],
    [NE, ZS, ZS, ZS, NS],
    [NE, UF, UCB, SS, DQN]
]
#%%
Game = PF(Strategy_sets[7])
Game.run(num_10k_rounds=1000, log=False)
Game.post_game_analysis()

# %%
