#%%
import matplotlib.pyplot as plt 
from common import PlatForm as PF
from strategies import RandomStrategy as RS
from strategies import HuristicStrategy as HS
from strategies import StatisticalStrategy as SS
from strategies import NashEquilibrium as NE 
from strategies import AdaptiveStrategy as AS 
from strategies import AdaptiveNasheqilibrium as AN 
from strategies import ReinforcementLearning as RL
from strategies import DealerStrategy as DS
l=[RS(1,False),RS(3,False),RS(0,False),RS(2,True),NE(0.97),NE(),AN(5000),
RL(0,exploration=0.1,init_reward=3),
RL(1,exploration=0.1,c=3,init_reward=2),
RL(2,a=0.1,baseline=2)]
l+=[AS(girdsize=200,discount=1,init_w=1,pt_threshold=2500,lowbd_threshold=1500,cooldown=10000)]
l+=[DS()]
Game=PF(l,num_rounds=1000000)
L=Game.run()
print(L[0])



#%%
L[1].plot()
L[1].plot(ylim=(0.8,1))
L[1].plot(ylim=(0.9,1))
L[1].plot(ylim=(0.93,0.96))
plt.show()
#%%
print(l[-3].bandits_rewards)
#%%


