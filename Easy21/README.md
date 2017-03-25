This assignment is from the reinforcement learing course in UCL by taught David Silver.
The problem statement is in the pdf file Easy21.
We apply reinforcement learning methods to a simple card game called Easy21.
This is similar to the Blackjack game but the rules are different.

Here are some figures we get from the results of Monte-Carlo control, TD learning and linear function approximation
on this game.

Here is the optimal value-action function we computed by Monte-Carlo control.
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Easy21/optimal-Q1.png)
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Easy21/optimal-Q2.png)

This is the the mean-squared error against distinct lambdas.
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Easy21/lambda-errors.png)

For lambda = 0 and lambda = 1, we plot the learning curve of mean-squared error against episode number.
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Easy21/learning-curve.png)

The following are the similar figures, but for TD learning by linear function approximation.
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Easy21/lambda-errors-FA.png)
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Easy21/learning-curve-FA.png)
