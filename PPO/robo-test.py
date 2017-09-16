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
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(num_inputs, 256)
        self.fc4 = nn.Linear(256, 128)

        self.critic_linear = nn.Linear(128, 1)
        self.actor_mean = nn.Linear(128, num_outputs)

        init_weight(self.fc1, 'relu')
        init_weight(self.fc2, 'relu')
        init_weight(self.fc3, 'relu')
        init_weight(self.fc4, 'relu')
        init_weight(self.critic_linear, 'linear')
        init_weight(self.actor_mean, 'tanh')
        #self.actor_mean.bias.data.mul_(0.0)
        #self.actor_mean.weight.data.mul_(0.1)
        #self.critic_linear.bias.data.mul_(0.0)
        #self.critic_linear.weight.data.mul_(0.1)
        self.actor_log_var = nn.Parameter(torch.zeros(1, num_outputs)-1.5)

        self.train()

    def act(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        mean = F.tanh(self.actor_mean(x))
        log_var = self.actor_log_var.expand_as(mean)
        variance = torch.exp(log_var)

        return mean, variance


    def critic(self, inputs):
        y = F.relu(self.fc3(inputs))
        y = F.relu(self.fc4(y))

        return self.critic_linear(y)

def process_obs(obs):
    # process observation 
    return torch.clamp(torch.Tensor(obs)*0.5, -1., 1.)



#from OpenGL import GLU
env = gym.make('RoboschoolHumanoid-v1')
obs = env.reset()
model = Net(44, 17)
model.load_state_dict(torch.load('weights3-humanoid', map_location=lambda storage, loc:storage))
#for _ in range(1000):
total = 0
while True:
    #env.render()
    state = process_obs(obs).unsqueeze(0)
    state = Variable(state, volatile=True)
    mu, sgm = model.act(state)
    action = to_numpy(mu, False)
    obs, reward, done, info = env.step(action[0])
    total += reward
    if done:
        obs = env.reset()
        print (total)
        total = 0

