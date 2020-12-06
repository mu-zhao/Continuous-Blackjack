
#%%
from scipy.integrate import quad 
from itertools import combinations
import numpy as np


#----------------------------------------------------------------------
def f(x,i,j):
    return (1-(1-x)*np.exp(x))**(i-j)*(2-np.e-x+np.exp(x))**j
def f1(x):
    return 1-(1-x)*np.exp(x)
def I(A,i,j):
    return f(A,i,j)-quad(f,A,1,args=(i,j))[0]

def solution(n,error=10**(-8)):
    res=np.zeros((n,n))
    for i in range(1,n):
        for j in range(i+1):
            h,l=1,0
            while h-l>error:
                m=(h+l)/2
                if I(m,i,j)>0:
                    h=m
                else:
                    l=m
            res[n-i-1,j]=l
    return res
#-------------------------------------------------------------------------------------
class RandomStrategy:
    """this strategy is for testing
    """
    def __init__(self,random_type,switch=True):
        self.algo_id='random strategy '+ str(random_type) +' with ' if switch else ' without '+'switch'
        self.type=random_type
        self.count=0
        self.switch=switch
        
    def para(self,i,num_players):
        self.num=num_players
        self.p=0
        return self
    def calibration(self,i,order,history,result,turn_reward):
        if self.type==0:
            self.p=np.random.sample()
        elif self.type==1:
            self.p=np.random.uniform(max(result),1)
        elif self.type==2 and self.count==0:
            self.p=0
        elif self.type==3:
            self.p=0.5
        if self.switch and self.count%100000==0:
            self.type=np.random.randint(4)
        self.count+=1
    def decision(self,hand):
        if hand<self.p:
            return True
        return False

#---------------------------------------------------------------------------------------------------------
class NaiveStrategy:
    """ Used for testing, this strategy can also be used as bench mark
    """
    def __init__(self):
        pass 
    def para(self,i,num_players):
        self.p=0
        return self
    def calibration(self,i,order,history,result,turn_reward):
        self.p=max(result)
    def decision(self,hand):
        if hand<self.p:
            return True
        return False

#-------------------------------------------------------------------------------------------------------------     



#-------------------------------------------------------------------------------------------------------------




#----------------------------------------------------------------------------------------------------------------



class NashEquilibrium:
    """ Nash equilibrium scaled by factor
    """
    def __init__(self,factor=1):
        self.factor=factor
    def para(self,i,num_players):
        self.threshold=solution(num_players)[:,0]*self.factor
        return self 
    def calibration(self,i,order,history,result,turn_reward):
        self.p=max(max(result),self.threshold[i])
    def decision(self,hand):
        if hand<self.p:
            return True
        return False

#--------------------------------------------------------------------------------------------------------
class AdaptiveNasheqilibrium:
    """ This strategy make the assumption that a player either plays Nash Equilibrium 
    or play a threshold uniformly chosen in $[0,1]$. More specifically, if rationality 
    assumption is violated, the algorithm will mark it as uniform threshold player.
    """
    def __init__(self,confidence_level=1500):
        self.confidence_level=confidence_level
        self.count=0
    def para(self,i,num_players):
        self.num_players=num_players
        self.id=i
        self.threshold=solution(num_players)
        self.profiles=np.zeros(num_players,dtype=int)
        self.record=[]
        return self 
    def calibration(self,rank,order,history,result,turn_reward):
        self.count+=1
        if len(history)>0:
            his_order,his_res,his_cards=history[-1][1:]
            for i in range(1,self.num_players):
                if his_order[i]!=self.id:
                    if max(his_cards[i])>=max(his_res[:i]):
                        self.profiles[his_order[i]]+=1
                    else:
                        self.profiles[his_order[i]]=-self.confidence_level
                        #self.record.append((self.count,i,his_order[i],his_cards,his_res))
        num_navie=0
        for i in range(rank+1,self.num_players):
            if self.profiles[order[i]]<0:
                num_navie+=1
        self.p=max(self.threshold[rank,num_navie],max(result))
    def decision(self,hand):
        if hand<self.p:
            return True
        return False
            

#-----------------------------------------------------------------------------------------------------------
class ModelfreeStrategy:
    """ This strategy is to estimate $L$ as opposed to $K$
        the estimate of $L(t)$ for small $t$ is not as accurate as for that of larger $t$ 
        since there is less chance for the previous score to be small. We can fix that to some extent by extrapolation.
        however the estimate of $L$ for small $t$ does not matter
        too much as the max is likely to be obtained for larger $t$ where the estimates are relatively accurate.
    """
    def __init__(self,size=1000,initial_count=2,extrapolation_decay_factor=0.8,extrapolation_range=5):
        self.m=size # discretization size 
        self.init_count=initial_count
        self.decay_factor=0.9
        self.e=np.exp(np.arange(self.m)/self.m)
        self.range=extrapolation_range
        self.xp=extrapolation_decay_factor**abs(np.arange(-extrapolation_range,extrapolation_range+1))
    def para(self,i,num_players):
        self.num_players=num_players
        self.id=i
        self.P={}# profiles
        for j in range(num_players):
            if j!=self.id:
                self.P[j]=[np.zeros((num_players,self.m)),np.zeros((num_players,self.m))+self.init_count]
                #self.P[j][:]=self.g 
        return self 
    def calibration(self,rank,order,history,result,turn_reward):
        if len(history)>0:
            position,scores=history[-1][1:3]# only need last turn's results for positions and scores
            t=0
            for i in range(self.num_players):
                player_id,T=position[i],int(t*self.m)
                if position[i]!=self.id:
                    if T<self.m/3:
                        l,h=max(0,T-self.range),T+self.range+1
                        self.P[player_id][1][i,l:h]+=self.xp[l-T+self.range:h-T+self.range]
                        self.P[player_id][0][i,l:h]+=((scores[i]<t)-self.P[player_id][0][i,l:h])/self.P[player_id][1][i,l:h]
                    else:
                        self.P[player_id][1][i,T]+=1
                        self.P[player_id][0][i,T]+=((scores[i]<t)-self.P[player_id][0][i,T])/self.P[player_id][1][i,T]
                t=max(t,scores[i])
        self.p=max(result)
        if rank<self.num_players-1:
            M=np.ones(self.m)
            for j in range(rank+1,self.num_players):
                M*=self.P[order[j]][0][j]
            A=np.argmax(np.cumsum(M[::-1])[::-1]*self.e)+np.random.random_sample()
            self.p=max(self.p,A/self.m)
            
    
    def decision(self,hand):
        if hand<self.p:
            return True
        return False 
#------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------

class Profile:
    """ this is the profile for AdaptiveStrategy, which is families of distributions.
       To enhance the performance, I run two tests: lower bound assumption and point strategy assumption.
       lower bound means rationality assumption holds,i.e. the threshold will be greater than the max previous score
       point bound means the opponents' strategies are not randomized, that is, it's always a single number. 
    """
    def __init__(self,i,num_players,G,init_w,gridsize=1000,discount=0.99,pt_threshold=2500,lowbd_threshold=1500,cooldown=10000):
        self.Id=i 
        self.pt_threshold=pt_threshold
        self.lowbd_threshold=lowbd_threshold
        self.grid=gridsize
        self.discount=discount
        self.G=G
        self.weight=[np.zeros(self.grid)+init_w for _ in range(num_players)]
        self.dist=[np.zeros((self.grid,self.grid))+1/self.grid for _ in range(num_players)]
        self.init_sh=np.sum(self.G,axis=1)/self.grid
        self.sh=[np.copy(self.init_sh) for _ in range(num_players)]
        a=np.zeros((self.grid,2))
        a[:,1]=1
        self.bounds=[np.copy(a) for _ in range(num_players)]
        self.lowbd_established=[False]*num_players 
        self.pt_established=[False]*num_players
        self.acceptable_error=2/self.grid
        self.lowbd_count=np.zeros(num_players,dtype=int)
        self.pt_count=np.zeros(num_players,dtype=int)
        self.cooldownturns=cooldown 
        self.cooldown=3*cooldown 
        #self.record=[]
        
    

    def update(self,rank,t,a,b):
        self.cooldown-=1
        if t<=b:
            self.lowbd_count[rank]+=1
            if not (self.lowbd_established[rank] or self.pt_established[rank]) and self.lowbd_count[rank]>self.lowbd_threshold:
                self.establish_lowbd(rank)
        elif self.lowbd_established[rank]:
                self.lowbd_breached(rank)
        A=int(self.grid*t)
        l,h=self.bounds[rank][A]
        if a<h+self.acceptable_error and b>l-self.acceptable_error:
            self.pt_count[rank]+=1
            self.bounds[rank][A]=max(l,a),min(b,h)
            if not self.pt_established[rank] and self.pt_count[rank]>self.pt_threshold:
                self.establish_pt(rank)
        elif self.pt_established[rank]:
            self.pt_breached(rank)
        x=int(a*self.grid)
        y=min(int(self.grid*b)+1,self.grid)
        if self.pt_established:
            self.pt_fit(rank,A)
        elif self.lowbd_established:
            self.lowbd_fit(rank,A,y)
        else:
            for k in range(-3,4):
                A1,x1,y1=A+k,x+k,y+k
                if 0<=A1<self.grid and 0<=x1 and y1<self.grid:
                    self.fit(rank,A1,x1,y1,0.9**abs(k))
        if self.cooldown>0:
            return True
        else:
            return False 

    def lowbd_breached(self,rank):
        self.lowbd_established[rank]=False 
        self.lowbd_count[rank]=0
        self.sh[rank][:]=self.init_sh
        self.dist[rank][:]=1/self.grid
        self.cooldown=self.cooldownturns
        #self.record.append(-1)
        
    def establish_lowbd(self,rank):
        self.lowbd_established[rank]=True 
        for A in range(self.grid):
            z=sum(self.dist[rank][A,A:])
            self.dist[rank][A,:A]=0
            self.dist[rank][A][A:]/=z
            self.sh[rank][A]=sum(self.dist[rank][A,A:]*self.G[A,A:])
        #self.record.append(1)

    def establish_pt(self,rank):
        self.pt_established[rank]=True 
        for A in range(self.grid):
            self.pt_fit(rank,A)
       # self.record.append(2)

    def pt_breached(self,rank):
        self.pt_established[rank]=False
        #self.record.append((-2,self.bounds[rank]))
        self.bounds[rank][:0]=0
        self.bounds[rank][:1]=1
        self.pt_count[rank]=0
        self.sh[rank]=self.init_sh
        self.dist[rank][:]=1/self.grid
        self.cooldown=self.cooldownturns
        

    def pt_fit(self,rank,A):
        l,h=self.bounds[rank][A]
        x=max(0,int((l-self.acceptable_error)*self.grid))
        y=min(int((self.acceptable_error+h)*self.grid)+1,self.grid)
        self.dist[rank][A][:]=0
        self.dist[rank][A][x:y]=1/(y-x)
        self.sh[rank][A]=sum(self.init_sh[x:y])*self.grid/(y-x)

    def lowbd_fit(self,rank,A,y):
        self.fit(rank,A,A,y)

    def fit(self,rank,A,x,y,extraplation_factor=1):
        Z=sum(self.dist[rank][A,x:y])*self.discount
        w=np.zeros(self.grid)+self.weight[rank][A]
        delta=extraplation_factor*sum(self.dist[rank][A][x:y]*self.G[A][x:y])/Z
        self.sh[rank][A]*=self.weight[rank][A]
        self.sh[rank][A]+=delta
        self.weight[rank][A]+=extraplation_factor/self.discount
        self.sh[rank][A]/=self.weight[rank][A]
        w[x:y]+=extraplation_factor/Z
        w/=self.weight[rank][A]
        self.dist[rank][A]*=w 

    def getsh(self,rank):
        return self.sh[rank]
    def getprofile(self,rank):
        return self.dist[rank]
    
        

class AdaptiveStrategy:
    """This was the model free strategy,  made before I 
        realized that the conditions can be imposed on $L$ instead of the strategy $K$. 
    """
    def __init__(self,girdsize=100,discount=0.9,init_w=10,pt_threshold=1500,lowbd_threshold=1000,cooldown=4000):
        self.grid=girdsize 
        self.discount=discount
        self.init_w=init_w
        self.cooldown=cooldown
        self.pt_threshold=pt_threshold
        self.lowbd_threshold=lowbd_threshold
        temp=(np.arange(self.grid)+0.1)/self.grid
        self.exp_temp=np.exp(temp)
        res=1-(1-temp)*self.exp_temp 
        self.G=np.zeros((self.grid,self.grid))
        for t in range(self.grid):
            self.G[t]=res
            self.G[t,:t]=(t/self.grid-temp[:t])*self.exp_temp[:t]
    def para(self,Id,num_players):
        self.Id=Id  
        self.nashequilibrum=solution(num_players)
        self.profiles=[Profile(i,num_players,self.G,self.init_w,self.grid,self.discount,self.pt_threshold,self.lowbd_threshold,self.cooldown) for i in range(num_players)]
        self.profiles[Id]=None 
        self.num_players=num_players
        return self 
    def calibration(self,i,order,history,result,turn_reward):
        self.iscooldown=False 
        if len(history)>0:
            od,res,cards=history[-1][1:]
            for j in range(self.num_players):
                if od[j]!=self.Id:
                    maxindex=np.argmax(cards[j])
                    a=cards[j,maxindex-1]
                    b=min(1,cards[j,maxindex])
                    if j==0:
                        t=0
                    else:
                        t=max(res[:j])
                    if self.profiles[od[j]].update(j,t,a,b):
                        self.iscooldown=True
        if self.iscooldown:
            num=0
            for j in range(i+1,self.num_players):
                if not ( (self.profiles[order[j]].pt_established) or (self.profiles[order[j]].lowbd_established)):
                    num+=1
            self.p=max(max(result),self.nashequilibrum[i,num])
        elif i==self.num_players-1:
            self.p=max(result)
        else:
            W=np.ones(self.grid)
            for j in range(i+1,self.num_players):
                W*=self.profiles[order[j]].getsh(j)
            self.p=np.argmax(np.cumsum(W[::-1])[::-1]*self.exp_temp)+0.5
            self.p/=self.grid
            self.p=max(self.p,max(result))
        

    def decision(self,hand):
        if hand<self.p:
            return True
        return False
    


        

#--------------------------------------------------------------------------------------------------------------
    




class ContextualBandits:
    """ This is the classic algorithm for k-armed bandit problem, with some modifications to improve the performance.
    """
    def __init__(self,rl_type,resource_limit=3,size=1000,exploration=0.15,
    init_reward=2,xp_dacay_rate=0.9,a=0.1,c=2,baseline=4):
        self.resource_limit=resource_limit # how many positions into the rank we want to record
        self.count=0
        self.type=rl_type
        self.last_choice=None
        self.exploration=exploration
        self.xp_decay=xp_dacay_rate
        self.a=a # learning rate for gradient descent
        self.baseline=baseline 
        self.init_reward=init_reward 
        self.m=size 
        self.c=c
        
    def para(self,i,num_players):
        self.id=i 
        self.num_players=num_players
        self.dic={}
        self.N=self.code()
        self.bandits_rewards=np.zeros((self.N,self.m))+self.init_reward # row major for our purpose for speed
        self.bandits_num=np.zeros((self.N,self.m),dtype=int)+1
        if self.type==2:
            self.H=np.zeros((self.N,self.m))+self.baseline
            self.dist=np.zeros((self.N,self.m))+1/self.m
        return self 

    def calibration(self,rank,order,history,result,turn_reward):
        self.count+=1
        if self.count%(self.m*self.N*20)==0:
            self.exploration*=self.xp_decay
        if self.last_choice:
            winner,od=history[-1][:2]
            i,c=self.last_choice
            r=(od[winner]==self.id)
            self.bandits_rewards[i,c]+=(r-self.bandits_rewards[i,c])/self.bandits_num[i,c]
            if self.type==2:
                self.H[i]-=self.a*(r-self.bandits_rewards[i][c]*self.dist[i])
                self.H[i][c]+=self.a*(r-self.bandits_rewards[i][c])
            self.last_choice=None 
        self.p=max(result)
        if rank!=self.num_players-1: 
            if rank<self.resource_limit:
                row=self.dic[(tuple(sorted(order[:rank])),0)]
            elif rank>=self.num_players-self.resource_limit:
                row=self.dic[(tuple(sorted(order[rank+1:])),1)]
            else:
                row=self.dic[rank]
            if self.type==0:
                self.greedy(row)
            elif self.type==1:
                self.UCB(row)
            else:
                self.gradient(row)
                   
    def decision(self,hand):
        if hand<self.p:
            return True
        return False

    def UCB(self,row):
        n=int(self.p*self.m)
        choice=n+np.argmax(self.bandits_rewards[row][n:]+self.c*np.sqrt(np.log(self.count)/self.bandits_num[row][n:]))
        self.p=max(self.p,(choice+np.random.sample())/self.m)
        self.last_choice=(row,choice)
        self.bandits_num[row,choice]+=1


    def greedy(self,row):

        n=int(self.p*self.m)
        if np.random.sample()<self.exploration:
           # choice=n+np.argmin(self.bandits_num[row][n:])
            choice=np.random.randint(n,self.m)
        else:
            choice=n+np.argmax(self.bandits_rewards[row][n:])
        self.p=max(self.p,(choice+np.random.sample())/self.m)
        self.last_choice=(row,choice)
        self.bandits_num[row,choice]+=1
        

    def gradient(self,row):
        n=int(self.p*self.m)
        self.dist[row]=np.exp(self.H[row])
        self.dist[row]/=sum(self.dist[row])
        choice=n+np.random.choice(self.m-n,1,p=self.dist[row][n:]/sum(self.dist[row][n:]))
        self.p=max(self.p,(choice+np.random.sample())/self.m)
        self.last_choice=(row,choice)
        self.bandits_num[row,choice]+=1

    def code(self):
        k=0
        for j in range(self.resource_limit,self.num_players-self.resource_limit):
            self.dic[j]=k
            k+=1
        L=[i for i in range(self.num_players) if i!=self.id]
        for j in range(self.resource_limit):
            for c in combinations(L,j):
                for d in range(2):
                    self.dic[(c,d)]=k
                    k+=1 
        return k 
    def diagnosis(self):
        if self.type==2:
            return self.H, self.dist, self.bandits_num, self.bandits_rewards
        return self.bandits_num, self.bandits_rewards
    
#---------------------------------------------------------------------------------------------------------------
        
class RLwithPS:  
    def __init__(self,algo_type,size=100,resource_limit=3,xp_rate=0.15,
                 xp_decay=0.9,initial_reward=1,ucb_sigma=2,gradient_learning_rate=0.05,baseline=3,num_split_and_dump=8):
        self.m=size
        self.limit=resource_limit
        self.type=algo_type
        self.xp=xp_rate
        self.decay=xp_decay
        self.init_r=initial_reward
        self.c=ucb_sigma
        self.a=gradient_learning_rate
        self.baseline=baseline 
        self.last_choice=None 
        self.num_sd=num_split_and_dump
    def para(self,i,num_players):
        self.id=i
        self.num=num_players
        self.dic={}
        self.N=self.code() #total number of bandits
        self.Bxp=np.zeros(self.N)+self.xp #exploration rate for each bandit
        self.M=np.zeros((self.N,4,self.m))# arm reward,count, interval left endpoint ,interval length
        self.M[:,0,self.m//4:]=self.init_r #initialize reward
        self.M[:,1,:]=5 # initialize count
        self.M[:,2,:]=np.arange(self.m)/self.m #initialize interval left endpoint
        self.M[:,3,:]=1/self.m #initialize interval length
        self.Bcount=np.zeros(self.N) # count for each bandit
        self.Bsd=np.ones((self.N,2),dtype=int)
        self.Bsd[:,1]=self.m*500
        if self.type==2:
            self.G=np.zeros((self.N,2,self.m))
            self.G[:,0,:]=self.baseline # preference
            self.G[:,1,:]=1/self.m   # probability
        return self 
    def calibration(self,rank,order,history,result,turn_reward):
        if self.last_choice:
            winner,od=history[-1][:2]
            last_col,c=self.last_choice
            r=(od[winner]==self.id)
            self.M[last_col,0,c]+=(r-self.M[last_col,0,c])/self.M[last_col,1,c]
            if self.type==2:
                self.G[last_col,0,:]-=self.a*(r-self.M[last_col,0,c]*self.G[last_col,1])
                self.G[last_col,0,c]+=self.a*(r-self.M[last_col,0,c])
            self.last_choice=None 
        self.p=max(result)
        if rank!=self.num-1: 
            if rank<self.limit:
                row=self.dic[(tuple(sorted(order[:rank])),0)]
            elif rank>=self.num-self.limit:
                row=self.dic[(tuple(sorted(order[rank+1:])),1)]
            else:
                row=self.dic[rank]
            if self.type!=2:
                self.exploration_exploitation(row)
            else:
                self.gradient(row)
    
    def exploration_exploitation(self,row):
        n=self.index(row)
        if n>=0:
            if self.type: # UCB
                choice=n+np.argmax(self.M[row,0,n:]+self.c*np.sqrt(np.log(self.Bcount[row])/self.M[row,1,n:]))
            else: # epsilon greedy
                ran_num=np.random.sample()
                if ran_num<self.Bxp[row]:
                    max_reward_position=np.argmax(self.M[row,0])
                    if n<max_reward_position: # n<max_reward positions are underexplored 
                        if ran_num<self.Bxp[row]/3:
                            choice=n+np.argmin(self.M[row,1,n:max_reward_position]) # this is compensation.
                        else:
                            choice=np.random.randint(n,max_reward_position) #exploitation
                    else:
                        choice=n
                else: # expolitation 
                    choice=n+np.argmax(self.M[row,0,n:])
            self.p=max(self.p,self.M[row,2,choice]+self.M[row,3,choice]*np.random.sample())
            self.last_choice=(row,choice)
            self.M[row,1,choice]+=1
            self.Bcount[row]+=1
            self.Bsd[row,1]-=1
            if self.Bsd[row,1]==0:
                self.policy_pruning_and_shrinking(row)
            if self.type==0 and self.Bsd[row,0]==self.num_sd and self.Bcount[row]%(100*self.m)==0:
                self.Bxp[row]*=self.decay
    
    def gradient(self,row):
        n=self.index(row)
        if n>=0:
            self.G[row,1]=np.exp(self.G[row,0])
            self.G[row,1]/=sum(self.G[row,1])
            choice=n+np.random.choice(self.m-n,1,p=self.G[row,1,n:]/sum(self.G[row,1,n:]))
            self.p=max(self.p,(choice+np.random.sample())/self.m)
            self.p=max(self.p,self.M[row,2,choice]+self.M[row,3,choice]*np.random.sample())
            self.last_choice=(row,choice)
            self.M[row,1,choice]+=1
            self.Bsd[row,1]-=1
            if self.Bsd[row,1]==0:
                    self.policy_pruning_and_shrinking(row)
 
    def index(self,row):
        """ find which bucket any given real number is in.
            Could use binary search, not faster in this setting
        Args:
            row ([int]): [index of the bandit]

        Returns:
            [int]: [index of the arm]
        """
        if self.p<self.M[row,2,1]:
            return 0
        if self.p>=sum(self.M[row,2:,-1]):
            return -1
        return np.argmin(self.p>=self.M[row,2])
    
    def policy_pruning_and_shrinking(self,row):
        
        """only dump the leftmost and rightmost arms in the bottom 20%

        Args:
            row ([int]): [the index of the bandit]
        """
        
        performance=self.M[row,0].argsort() # rank of the arms by reward
        least_chosen=(self.M[row,1]).argsort()
        print(least_chosen[:self.m//10])
        print('---------------------')
        print(performance[:self.m//5])
        s=np.zeros(self.m,dtype=int)
        s[performance[:self.m//5]]=1 #lowest 20% performance 
        s[least_chosen[:self.m//10]]=1 #lowset 10%
        under_perform=np.where(s==0)
        dump_left_limit=under_perform[0][0] #left of the limit will be dumped
        dump_right_limit=under_perform[0][-1]#right of the limit will be dumped
        print(row,self.Bsd[row,0],dump_left_limit,dump_right_limit)
        print('*************************')
        num_dump=dump_left_limit+self.m-dump_right_limit-1
        top_performance=[]
        for k in performance[::-1]:
            if dump_left_limit<=k<=dump_right_limit: # it could happend that for small $t$, 
                top_performance.append(k)           #  the performance is high because of sheer luck      
                if len(top_performance)>=num_dump:   # as there are few trails
                    break 
        if num_dump>0:
            temp=np.zeros((4,self.m)) # new armed bandit
            if self.type:
                temp_H=np.zeros(self.m)
            i=dump_left_limit
            l=0
            for j in sorted(top_performance):
                r=l+j-i 
                temp[:,l:r]=self.M[row,:,i:j]
                temp[0,r:r+2]=self.M[row,0,j]
                temp[1,r:r+2]=self.M[row,1,j]
                temp[3,r:r+2]=self.M[row,3,j]/2
                temp[2,r]=self.M[row,2,j]
                temp[2,r+1]=sum(temp[2:,r])
                if self.type==2:
                    temp_H[l:r]=self.G[row,0,i:j]
                    temp_H[r:r+2]=self.G[row,0,j]
                i,l=j+1,r+2
                
            temp[:,l:]=self.M[row,:,i:dump_right_limit+1]
            if self.type==2:
                temp_H[l:]=self.G[row,0,i:dump_right_limit+1]
                self.G[row,0]=temp_H
                self.G[row,1]=np.exp(temp_H)
                self.G[row,1]/=sum(self.G[row,1])
            self.M[row]=temp
            self.Bcount[row]=sum(self.M[row,1])
            self.Bsd[row,0]+=1
            if self.Bsd[row,0]==self.num_sd:
                self.Bsd[row,1]=-1
            else:
                self.Bsd[row,1]=self.m*100*self.Bsd[row,0]
        else:
            self.Bsd[row,1]=self.m*100

    def code(self):
        k=0
        for j in range(self.limit,self.num-self.limit):
            self.dic[j]=k
            k+=1
        L=[i for i in range(self.num) if i!=self.id]
        for j in range(self.limit):
            for c in combinations(L,j):
                for d in range(2):
                    self.dic[(c,d)]=k
                    k+=1 
        return k 
    def diagnosis(self):
        if self.type==2:
            return self.M, self.G 
        return self.M 

