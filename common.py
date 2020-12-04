import numpy as np
import pandas as pd 
import time 
class PlatForm:
    def __init__(self,players):
        self.count=0
        self.num_player=len(players)
        self.turn_reward=1
        self.rewards=np.zeros((self.num_player,self.num_player)) # cumulative rewards breakdown by position
        self.bet_result=np.zeros(self.num_player) # bet score of the current round
        self.bet_history=[] # all previous rounds but the current round
        self.players=[player.para(Id,self.num_player) for Id,player in enumerate(players) ]
        

    def run(self,num_thousand_rounds=10000):
        self.rounds=num_thousand_rounds*1000
        self.record=np.zeros((1000,self.num_player))
        start_time=time.time()
        for k in range(self.rounds):
            self.count+=1
            self.bet_result[:]=0
            # for all practical purposes, 16 suffices
            self.cards=np.cumsum(np.random.sample((self.num_player,16)),axis=1)
            self.order=np.random.permutation(self.num_player) #reshuffle of players
            for i in range(self.num_player):
                self.deal(i)
            winner=np.argmax(self.bet_result)
            """
            #self.best_history is all previous history, for the efficiency we use part 2, since we only 
            # need to access the latest history
            #-----------------------------------------------------------------------
            # part 1
            #self.bet_history.append((winner,self.order,self.bet_result,self.cards))
            #-----------------------------------------------------------------------------
            # part 2
            """
            self.bet_history=[(winner,np.copy(self.order),np.copy(self.bet_result),np.copy(self.cards))]
            self.rewards[self.order[winner],winner]+=self.turn_reward
            if k%num_thousand_rounds==0:
                self.record[k//num_thousand_rounds][:]=np.sum(self.rewards[:self.num_player],axis=1)
                self.turn_reward+=1
                if k%1000000==0:
                    print('%s million rounds done, %s s elapsed'%(k//1000000,np.round(time.time()-start_time,4)))
            
    def deal(self,i):
        strategy=self.players[self.order[i]] 
        strategy.calibration(i,self.order,self.bet_history,self.bet_result,self.turn_reward)
        """
        # if the decision made does not depend on the sequence, to make it faster you can use the
        # part 2 lines instead
        #---------------------------------
        #part 1
        #k=0
        #while self.cards[i][k]<=1 and strategy.decision(self.cards[i][k]):
        #    k+=1
        #-----------------------------------------
        # part 2
        """
        k=np.argmax(self.cards[i]>strategy.p)
        #-------------------------------------------
        if self.cards[i][k]>1:
            self.bet_result[i]=0
        else:
            self.bet_result[i]=self.cards[i][k]
        self.cards[i][k+1:]=0

    def analysis(self):
        s=np.sum(self.rewards,axis=1)
        print('total reward:\n ',s/s[0])
        print('breakdown by rank:\n ',self.rewards/self.rewards[0])
        print('percentege breakdown: \n',self.rewards/s)
        return pd.DataFrame(self.record),self.rewards
        
        
        
        
            

            


    
