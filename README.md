# Continuous-Blackjack
Continuous Blackjack is a variant of the classic poker game blackjack. Here are the rules:
* The game will run for a large number of rounds, say, one billion.
* Each round every player's position is reshuffled randomly.
* From the first player, each player plays their turn in order, and other players can observe the previous players' actions.
* Each player can choose to hit or stay. If the player choose to hit,  a random number is generated form standard uniform distribution and added to the palyer's total sum; otherwise the player's trun ends.
* players' scores are the total sums as long as they don't exceeds $1$, in which case the score will be $0$. At the end of each round, the player with the highest score receive one point.
* In the rare scenario where two or more players get the same highest score, they will share the point equally among them.
--------------------------
### Files:
1. commom.py is the platform on which the game is played.
2. strategies.py is the file where the different strategies lie.
3. test.py is the file to test different strategies.
---------------------------
### Strategies:

The strategies here are based on my paper [Continuous Blackjack: Equilibrium, Deviation and Adaptive Strategy](https://arxiv.org/abs/2011.10315)

--------------------------

### Here's the template to write your own strategy

```Python 3
class YourStrategy:
    """this is the template to create your own strategy.
    """
    def __init__(self,customized_arg): # you can specify your strategy with any parameters
        pass 
    def para(self,i,num_players): # no customized_arg
        pass
        return self # must return self
    def calibration(self,i,order,history,result,turn_reward):
        pass # if you use threshold, make sure the attribute is p. Have a look at the strategies provided in strategies.py.
    def decision(self,hand): # in case your strategy depends on the sequence of the random numbers rather than the prefix sum.
        pass 
 ```
 ----

