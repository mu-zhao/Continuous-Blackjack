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
from strategies import ModelfreeStrategy as MS 
l=[RS(0,False),RS(1,False),RS(2,False),RS(3,False),NE(0.98),NE(),
MS(size=500,initial_count=1,extrapolation_decay_factor=0.6,extrapolation_range=4)]
#l+=[AS(girdsize=200,discount=1,init_w=1,pt_threshold=2500,lowbd_threshold=1500,cooldown=10000)]
#%%
Game=PF(l,num_rounds=25000000)
L=Game.run()
print(L[0])



#%%
L[1].plot()
#%%
L[1].plot(ylim=(0.165,0.175))

plt.show()
#%%



