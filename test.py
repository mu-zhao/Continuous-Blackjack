#%%
import matplotlib.pyplot as plt 
from common import PlatForm as PF
from strategies import RandomStrategy as RS
from strategies import NaiveStrategy as NS
from strategies import NashEquilibrium as NE 
from strategies import AdaptiveStrategy as AS 
from strategies import AdaptiveNasheqilibrium as AN 
from strategies import ReinforcementLearning as RL
from strategies import ModelfreeStrategy as MS 
#%%
#%%
l=[RS(0,False),RS(1,False),RS(2,False),RS(3,False),NS(),NE()]
#l+=[MS(size=500,initial_count=2,extrapolation_decay_factor=0.6,extrapolation_range=4)]
l+=[RL(size=1000,rl_type=0,exploration=0.05,init_reward=1,xp_dacay_rate=0.5,resource_limit=3)]
l+=[RL(size=1000,rl_type=1,exploration=0.05,init_reward=1,xp_dacay_rate=0.5,c=2,resource_limit=3)]
#l+=[RL(size=200,rl_type=2,a=0.05,baseline=3,resource_limit=4)]
#l+=[AS(girdsize=200,discount=1,init_w=1,pt_threshold=2500,lowbd_threshold=1500,cooldown=10000)]

Game=PF(l)
#%%
Game.run(num_thousand_rounds=50000)
L=Game.analysis()
L[0].plot()


#%%
L[0].plot(ylim=(0.1,0.16))







# %%
