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
l=[NE(),NE(),NE(),NE(),NE()]
#l+=[AS(girdsize=200,discount=1,init_w=1,pt_threshold=2500,lowbd_threshold=1500,cooldown=10000)]

Game=PF(l,num_rounds=100000)
L=Game.run()
print(L[0])



#%%
L[1].plot()
#%%
L[1].plot(ylim=(0.195,0.205))

plt.show()
#%%
print(l[-1].bandits_num)
#%%


