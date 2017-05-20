This small project is modified from the second assignment of the cource ''Deep Reinforcement Learning and Control'', 
Spring 2017, CMU 10703. 
We studied the following papers and implemented several different models for the Atari game ''Enduro'' based on the 
toolkit OpenAi gym.
1. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 
Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.
2. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, 
Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. 
Nature, 518(7540):529–533, 2015.
3. Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. 2016.
4. Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas. Dueling network 
architectures for deep reinforcement learning. arXiv preprint arXiv:1511.06581, 2015.

First, we review the above papers. All the papers apply some variant of Q-learning. 
The architecture introduced in [1, 2] is the deep Q-network, a convolutional network as the value-action function approximator.
The input consists of 4 (or 3, depending on the game) consecutive frames as the state of the MDP, and the output is a list of 
approximating Q values for all possible actions. The two important ideas introduced for training the network in the paper are 
target network and experience replay. 
- The target network can be seen as the old version of the current Q-network, which is updated to Q-network every k training 
steps. When training the Q-network, we use the values from target network as the targets
of the training. The reason for this is that if targets are also from our current trained Q-network which is always changing, 
the Q-values are not stable and the training may not converge. 
- Experience replay stores the last k transitions in a replay memory and sample a minibatch from this memory to train
the Q-network. The sample could be uniformly random, or we could use other method like prioritized replay (use a priority 
queue to store those transitions). It can increase the data efficiency, break the correlations between samples and reduce 
variance of the updates. It also motivates the choice of Q-learning: since the current Q-network is different from the 
previous one that generates the samples, we can only learn off-policy. 

The new idea of [3] is to decouple the selection and evaluation when compute the target value. The target value can be computed
by first finding an greedy action and then compute the Q-value of the greedy action. In [1, 2], these two operations are done
in the same network: the target network. This results in overoptimistic value estimations. To solve this problem, it is 
suggested in [3] to select the greedy action by the current Q-network, and then compute the Q-value by the target network.

In [4], a new architecture, called dueling network, is introduced. The previous deep Q-network directly approximates the 
Q-value for all actions, but the dueling network implicitly approximates the value function and advantage function for all
actions, and then combines them to obtain the Q-values for all actions. The training algorithm could be the same as [1, 2] 
or [3].

We implemented three kinds of architectures: a simple linear Q-network, the deep Q-network in [1, 2], the dueling network in [4].
We also implemented the training algorithm in [1, 2] and [3]. 

Note that games from openai gym is different from ALE (Arcade Learning Environment), the environment used in [1, 2]: the 
action will be repeated 2, 3, or 4 times randomly. So the preprocessing for openai games is a little different from that 
in the papers. 
The architecture introduced in [1, 2] is the deep Q-network, a convolutional network as the value-action function approximator.
