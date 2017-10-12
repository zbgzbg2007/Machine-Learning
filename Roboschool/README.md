We implement by pytorch three different algorithms, including A3C, PPO and a variant of DDPG, and compare their performances based on environments in Roboschool from OpenAI.


Here is the result for PPO and DDPG on Humanoid-v1:
![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/Roboschool/Humanoid.png)

The trained weights by PPO is in the file 'weights3-humanoid'.

It seems A3C does not learn anything in a few millions of steps, so we didn't try more on that.
