#%%
import matplotlib.pyplot as plt 
from common import PlatForm as PF
from strategies import RandomStrategy as RS
from strategies import NaiveStrategy as NS
from strategies import NashEquilibrium as NE 
from strategies import AdaptiveStrategy as AS 
from strategies import AdaptiveNasheqilibrium as AN 
from strategies import ContextualBandits as CB
from strategies import ModelfreeStrategy as MS 
from strategies import RLwithSD as RL 


#%%
l=[RS(0,False),RS(1,False),RS(2,False),RS(3,False),NS(),NE()]
#l+=[MS(size=500,initial_count=2,extrapolation_decay_factor=0.6,extrapolation_range=4)]
#l+=[CB(size=100,rl_type=0,exploration=0.15,init_reward=0,xp_dacay_rate=0.8,resource_limit=3)]
#l+=[CB(size=100,rl_type=1,exploration=0.15,init_reward=0,xp_dacay_rate=0.5,c=2,resource_limit=3)]
#l+=[CB(size=200,rl_type=2,a=0.05,baseline=3,resource_limit=3)]
l+=[RL(algo_type=0,size=100,resource_limit=3,xp_rate=0.1,xp_decay=0.8,ucb_sigma=2,gradient_learning_rate=0.03,baseline=1,num_split_and_dump=8,initial_reward=0.5)]
#l+=[RL(algo_type=1,size=100,resource_limit=3,xp_rate=0.1,xp_decay=0.75,ucb_sigma=2,gradient_learning_rate=0.03,baseline=1,num_split_and_dump=10,initial_reward=0.3)]
#l+=[RL(algo_type=2,size=100,resource_limit=3,xp_rate=0.1,xp_decay=0.7,ucb_sigma=2,gradient_learning_rate=0.03,baseline=1,num_split_and_dump=10,initial_reward=0.3)]
#l+=[AS(girdsize=200,discount=1,init_w=1,pt_threshold=2500,lowbd_threshold=1500,cooldown=10000)]
Game=PF(l)

#%%
for i in range(3):
    Game.run(num_thousand_rounds=1000)
    L=Game.analysis()
    L[0].plot(ylim=(0.189,0.191))
    plt.show()
    




#%%
L[0].plot(ylim=(0.15,0.18))







# %%
