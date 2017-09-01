import roboschool, gym
import argparse
import torch
import random
from multiprocessing import set_start_method
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.init as init
import torch.nn.functional as F
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')
import torch.autograd as autograd
from torch.autograd import Variable
import time
import os
import sys
from collections import deque


def to_numpy(x, gpu):
    # return numpy of variable
    return x.cpu().data.numpy() if gpu else x.data.numpy()

def to_Variable(x, gpu):
    return Variable(x).cuda() if gpu else Variable(x)
  
def to_tensor(x): 
    return torch.Tensor(x)

def init_weight(layer, nonlinear):
    # initialize weights
    init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain(nonlinear))
    init.constant(layer.bias, 0.) 

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        # num_inputs (int): size of observation
        # num_outputs (int): size of action
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 300)
        self.fc2 = nn.Linear(300, 600)
        self.fc3 = nn.Linear(num_inputs+num_outputs, 300)
        self.fc4 = nn.Linear(300, 600)

        self.critic_linear = nn.Linear(600, 1)
        self.actor_linear = nn.Linear(600, num_outputs)

        init_weight(self.fc1, 'relu')
        init_weight(self.fc2, 'relu')
        init_weight(self.fc3, 'relu')
        init_weight(self.fc4, 'relu')
        init_weight(self.critic_linear, 'linear')
        init_weight(self.actor_linear, 'tanh')

        self.train()
    def act(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        action = F.tanh(self.actor_linear(x))
        return action

def process_obs(obs):
    # process observation 
    return torch.Tensor(obs)


#from OpenGL import GLU
env = gym.make('RoboschoolAnt-v1')
obs = env.reset()
model = Net(28, 8)
model.load_state_dict(torch.load('weights-ant'))
#for _ in range(1000):
while True:
    env.render()
    state = process_obs(obs).unsqueeze(0)
    state = Variable(state, volatile=True)
    action = model.act(state)
    action = to_numpy(action, False)
    obs, reward, done, info = env.step(action[0])
    if done: obs = env.reset()
