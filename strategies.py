
#%%
from scipy.integrate import quad 
import numpy as np

#----------------------------------------------------------------------
def f(x,i,j):
    return (1-(1-x)*np.exp(x))**(i-j)*(2-np.e-x+np.exp(x))**j
def f1(x):
    return 1-(1-x)*np.exp(x)
def I(A,i,j):
    return f(A,i,j)-quad(f,A,1,args=(i,j))[0]

def solution(n,error=10**(-5)):
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
    def __init__(self,naivetype,switch=True):
        self.type=naivetype
        self.count=0
        self.switch=switch
    def para(self,i,num_players,rounds):
        self.num=num_players
        self.p=0
        return self
    def calibration(self,i,order,history,result,turn_reward):
        if self.type==0:
            self.p=np.random.sample()
        elif self.type==1:
            self.p=np.random.uniform(max(result),1)
        elif self.type==2:
            self.p=max(np.random.sample(),max(result))
        elif self.type==3:
            self.p=0.5
        if self.switch and self.count%100000==0:
            self.type=np.random.randint(4)
    def decision(self,hand):
        self.count+=1
        if hand<self.p:
            return True
        return False

#---------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------     

class HuristicStrategy:
    def __init__(self,naive=0.1):
        self.naive=naive
    def para(self,i,num_players,rounds):
        self.num_players=num_players
        return self 
        
    def calibration(self,i,order,history,result,turn_reward):
        self.p=max(max(result)+self.naive,0.5)
        self.p+=(1-self.p)*self.naive*np.random.sample()

    def decision(self,hand):
        if hand<self.p:
            return True
        return False

#-------------------------------------------------------------------------------------------------------------

class StatisticalStrategy:
    """ does not work really.
    """
    def __init__(self,rate):
        self.rate=rate 
        self.count=0
    def para(self,i,num_players,rounds):
        self.record=np.zeros(num_players)
        self.rounds=rounds 
        return self 
    def calibration(self,i,order,history,result,turn_reward):
        if self.count>0:
            w=history[-1][0]
            self.record[w]+=self.rate*(history[-1][2][w]-self.record[w])
        self.count+=1
        self.rate*=(1-1/(1+self.count))
        self.p=max(self.record[i],max(result))
    def decision(self,hand):
        if hand<self.p:
            return True
        return False


#----------------------------------------------------------------------------------------------------------------



class NashEquilibrium:
    """ Nash equilibrium scaled by factor
    """
    def __init__(self,factor=1):
        self.factor=factor
    def para(self,i,num_players,rounds):
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
    or play a threshold uniformly chosen in [0,1]
    """
    def __init__(self,confidence_level=1500):
        self.confidence_level=confidence_level
        self.count=0
    def para(self,i,num_players,rounds):
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
    """ This strategy is the most applicable one. 
    """
    def __init__(self,size=1000,initial_count=2,extrapolation_decay_factor=0.8,extrapolation_range=5):
        self.m=size # discretization size 
        self.init_count=initial_count
        self.decay_factor=0.9
        self.e=np.exp(np.arange(self.m)/self.m)
        self.range=extrapolation_range
        self.xp=extrapolation_decay_factor**abs(np.arange(-extrapolation_range,extrapolation_range+1))
    def para(self,i,num_players,rounds):
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


    

    
#----------------------------------------------------------------------------------------------------------------

class Profile:
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
    def para(self,Id,num_players,rounds):
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
    




class ReinforcementLearning:
    def __init__(self,rl_type,exploration=0.15,init_reward=2,c=2,a=0.1,baseline=4):
        self.count=0
        self.type=rl_type
        self.last_choice=None
        self.exploration=exploration
        self.c=c 
        self.a=a 
        self.baseline=baseline 
        self.init_reward=init_reward 


    def para(self,i,num_players,rounds):
        self.id=i 
        self.num_players=num_players
        self.threshold=L[-num_players:]
        self.bandits_rewards=np.zeros((self.num_players,31))+self.init_reward
        self.bandits_num=np.zeros((self.num_players,31),dtype=int)+1 
        if self.type==2:
            self.H=np.zeros((self.num_players,31))+self.baseline
            self.dist=np.zeros((self.num_players,31))+1/31
        return self 

    def calibration(self,rank,order,history,result,turn_reward):
        self.count+=1
        if self.count%10000==0:
            self.exploration*=0.95
        if self.last_choice:
            winner,od=history[-1][:2]
            i,c=self.last_choice
            if od[winner]==self.id:
                r=1
            else:
                r=0
            self.bandits_rewards[i,c]+=(r-self.bandits_rewards[i,c])/self.bandits_num[i,c]
            if self.type==2:
                self.H[i]-=self.a*(r-self.bandits_rewards[i][c]*self.dist[i])
                self.H[i][c]+=self.a*(r-self.bandits_rewards[i][c])
        self.p=max(result)
        if rank==self.num_players-1: 
            self.last_choice=None 
        else:
            if self.type==0:
                self.greedy(rank)
            elif self.type==1:
                self.UCB(rank)
            else:
                self.gradient(rank)
            new_p=self.threshold[rank]-0.0015*self.choice*0.9**rank
            if new_p>self.p:
                self.bandits_num[rank,self.choice]+=1
                self.last_choice=(rank,self.choice)
                self.p=new_p
            else:
                self.last_choice=None 
    
        
    def decision(self,hand):
        if hand<self.p:
            return True
        return False

    def UCB(self,rank):
        if np.random.sample()<self.exploration:
            self.choice=np.random.randint(31)
        else:
            self.choice=np.argmax(self.bandits_rewards[rank]+self.c*np.sqrt(np.log(self.count)/self.bandits_num[rank]))
   


    def greedy(self,rank):
        if np.random.sample()<self.exploration:
            self.choice=np.random.randint(31)
        else:
            self.choice=np.argmax(self.bandits_rewards[rank])
        

    def gradient(self,rank):
        self.dist[rank]=np.exp(self.H[rank])
        self.dist[rank]/=sum(self.dist[rank])
        self.choice=np.random.choice(31,1,p=self.dist[rank])
        

        
