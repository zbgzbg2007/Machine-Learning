"""Run Atari Environment with Agent.
   Shared network version.  
"""
import random
import gym
import threading
import keras
import agent
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input) 
from keras.models import Model
from agent import Agent
from scipy import ndimage

def create_model(input_shape, num_actions):  
    """Create the network model by Keras.  

    - Input
      - input_shape: tuple(int, int, int)
        The input shape
      - num_actions: int
        Number of possible actions. Defined by the gym environment.

    - Output
      - keras.models.Model
        The network model used by agent.
    """ 

    inputs = Input(shape = input_shape)
    
    x = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu')(inputs)
    x = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)

    # two outputs of the network: value function approximator and probability for actions
    val = Dense(units=1, activation='linear', name='val_output')(x)
    p = Dense(units=num_actions, activation='softmax', name='pi_output')(x)

    # the two outputs are concatenated 
    outputs = keras.layers.concatenate([val, p])
    model = Model(inputs=inputs, outputs=outputs)

    return model

# thread class
class myThread(threading.Thread):
    def __init__(self, env, in_dim, eps, eps_m, lr, network):
        threading.Thread.__init__(self)
        self.env = env
        self.in_dim = in_dim
        self.eps = eps
        self.eps_m = eps_m
        self.lr = lr
        self.network = network

    def run(self):
        env = gym.make(self.env)
        num_actions = env.action_space.n
        
        # the global tf graph is needed
        global graph
        with graph.as_default(): 
            agent = Agent(
                 network=self.network,
                 in_dim=self.in_dim,
                 num_actions=num_actions,
                 eps=self.eps, 
                 eps_min=self.eps_m,
                 gamma=0.99,
                 global_w=global_weights,
                 T=T,
                 T_max=T_max,
                 t_max=5,
                 learning_rate=self.lr
                 )
            # train the network
            agent.fit(env)
        
# setup parameters
game = 'Enduro-v0'
env = gym.make(game)
in_dim = [105*4, 80, 1]
model = create_model(in_dim, env.action_space.n) 
graph = tf.get_default_graph()
eps = [0.6,  0.5,   0.4,    0.3,   0.4,   0.3,   0.2,    0.8,   0.7,   0.2, 0.3, 0.4, 0.7, 0.75]
eps_m = [0.01,  0.5,   0.1,    0.1,   0.4,   0.15,   0.2,    0.5,   0.01,   0.2, 0.3, 0.01, 0.05, 0.3]
lr = [0.002, 0.001, 0.0025, 0.002, 0.002, 0.001, 0.0007, 0.003, 0.0025, 0.0025, 0.001, 0.0015, 0.003, 0.002] 
T_max = 10000000
T = 0

gw = model.get_weights()

# global agent
agent = Agent(
         network=model,
         in_dim=in_dim,
         num_actions=env.action_space.n,
         eps=0.05,
         eps_min=0.05,
         gamma=0.99,
         global_w=gw,
         T=T,
         T_max=T_max,
         t_max=5,
         learning_rate=0.00001
         )
agent.compile()
#agent.fit(env)
# weights of the global agent
global_weights = agent.network.get_weights()
    

# run all the threads
threads = []
for e, em, l in zip(eps, eps_m, lr):
    thread = myThread(game, in_dim, e, em, l, agent.network)
    thread.start()
    threads.append(thread)

# wait until all threads terminates
for t in threads:
    t.join()

# set up the weights of global agent and evaluate it
agent.network.set_weights(global_weights)
print 'evaluation total rewards: ',  agent.evaluate(env, 4)

# save weights
agent.network.save_weights('trained_weights')
    
print 'main thread exit'

