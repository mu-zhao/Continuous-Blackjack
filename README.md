# Continuous-Blackjack

Continuous Blackjack is a variant of the classic poker game blackjack. Here are the rules:

* The game will run for many rounds, say, one billion.
* Each round every player's position is reshuffled randomly.
* From the first player, each player plays their turn in order, and other players can observe the previous players' actions.
* Each player can choose to hit or stay. If the player choose to hit, a random number is generated form standard uniform distribution and added to the player's total sum; otherwise the player's turn ends.
* Players' scores are the total sums as long as they don't exceed $1$, in which case the score will be $0$. At the end of each round, the player with the highest score receive one point.
* In the rare scenario where two or more players get the same highest score, they will share the point equally among them.

---------------

## Strategies

The strategies here are based on my paper [Continuous Blackjack: Equilibrium, Deviation and Adaptive Strategy](https://arxiv.org/abs/2011.10315)

---------------
