We implement three different algorithms, including A3C, PPO and a variant of DDPG, and compare their performances based on environments in Roboschool from OpenAI.


Here is the PPO result for Humanoid-v1.
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Roboschool/PPO-result.png)

The variant of DDPG can learn slowly for Humanoid-v1. 
After trained on the variant of DDPG in about 20 millions of steps, the agent can reach about 120 points.
It seems A3C does not learn anything in a few millions of steps, so we didn't try more on that.
