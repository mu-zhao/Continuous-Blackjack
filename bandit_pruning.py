
import numpy as np
from common_utils import BaseStrategy
class CBPruning(BaseStrategy):
    # TODO: need reorg currently not working.
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
                choice=n+np.argmax(self.M[row,0,n:]+self.c*np.sqrt(np.log(self.Bcount[row])/self.M[row,1,n:])) # can be reduced to product without computational expensive operations, but not necessary
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
        return np.argmin(self.p>=self.M[row,2]) # binary search is theoretically better, but not necessary
    
    def policy_pruning_and_shrinking(self,row):
        
        """only dump the leftmost and rightmost arms in the bottom 20%

        Args:
            row ([int]): [the index of the bandit]
        """
        
        performance=self.M[row,0].argsort() # rank of the arms by reward
        least_chosen=(self.M[row,1]).argsort()
        #print(least_chosen[:self.m//10])
        #print('---------------------')
        #print(performance[:self.m//5])
        s=np.zeros(self.m,dtype=int)
        s[performance[:self.m//5]]=1 #lowest 20% performance 
        s[least_chosen[:self.m//10]]=1 #lowset 10%
        under_perform=np.where(s==0)
        dump_left_limit=under_perform[0][0] #left of the limit will be dumped
        dump_right_limit=under_perform[0][-1]#right of the limit will be dumped
       # print(row,self.Bsd[row,0],dump_left_limit,dump_right_limit)
       # print('*************************')
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

