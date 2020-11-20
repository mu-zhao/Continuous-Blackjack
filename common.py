import numpy as np
import pandas as pd 
import time 
class PlatForm:
    def __init__(self,players,num_rounds=100000):
        self.num_player=len(players)
        self.rounds=num_rounds
        self.turn_reward=1
        self.rewards=np.zeros(self.num_player) # cumulative rewards
        self.bet_result=np.zeros(self.num_player) # bet score of the current round
        self.bet_history=[] # all previous rounds but the current round
        self.players=[player.para(Id,self.num_player,self.rounds) for Id,player in enumerate(players) ]
        
    def run(self):
        start_time=time.time()
        record=np.zeros((self.rounds//10000,self.num_player)) #just for the graph
        for k in range(self.rounds):
            self.bet_result[:]=0
            # for all practical purposes, 16 suffices
            self.cards=np.cumsum(np.random.sample((self.num_player,16)),axis=1)
            self.order=np.random.permutation(self.num_player)
            for i in range(self.num_player):
                self.deal(i)
            winner=np.argmax(self.bet_result)
            #self.best_history is all previous history, for the efficiency we use part 2, since we only 
            # need to access the latest history
            #-----------------------------------------------------------------------
            # part 1
            #self.bet_history.append((winner,self.order,self.bet_result,self.cards))
            #-----------------------------------------------------------------------------
            # part 2
            self.bet_history=[(winner,np.copy(self.order),np.copy(self.bet_result),np.copy(self.cards))]
            self.rewards[self.order[winner]]+=self.turn_reward
            if k%10000==0:
                record[k//10000][:]=self.rewards[:self.num_player]/k
                #if k%10000==0:
                #    print('%s rounds done, %s s elapsed'%(k,np.round(time.time()-start_time,4)))
        return self.rewards[:self.num_player]/self.rounds, pd.DataFrame(record)
            
    def deal(self,i):
        strategy=self.players[self.order[i]] 
        strategy.calibration(i,self.order,self.bet_history,self.bet_result,self.turn_reward)
        # if the decision made does not depend on the sequence, to make it faster you can use the
        # part 2 lines instead
        #---------------------------------
        #part 1
        #k=0
        #while self.cards[i][k]<=1 and strategy.decision(self.cards[i][k]):
        #    k+=1
        #-----------------------------------------
        # part 2
        k=np.argmax(self.cards[i]>strategy.p)
        #-------------------------------------------
        if self.cards[i][k]>1:
            self.bet_result[i]=0
        else:
            self.bet_result[i]=self.cards[i][k]
        self.cards[i][k+1:]=0

    def log(self):
        return self.bet_history
        
        
        
            

            


    
