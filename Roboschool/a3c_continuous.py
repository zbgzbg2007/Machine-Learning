# mostly follow ppo architecture
# two networks do not share parameters
# no entropy added
# no LSTM layer in networks
# variance is a parameter, not an output of the network

import gym, roboschool
import argparse
import torch
import random
import scipy.misc
from multiprocessing import set_start_method
import torch.nn as nn
import numpy as np
import torch.optim as optim
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
        self.fc1 = nn.Linear(num_inputs, 64) 
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(num_inputs, 64) 
        self.fc4 = nn.Linear(64, 64) 

        self.critic_linear = nn.Linear(64, 1)
        self.actor_mean = nn.Linear(64, num_outputs)

        init_weight(self.fc1, 'relu')
        init_weight(self.fc2, 'relu')
        init_weight(self.fc3, 'relu')
        init_weight(self.fc4, 'relu')
        init_weight(self.critic_linear, 'linear')
        init_weight(self.actor_mean, 'tanh')
        self.actor_mean.bias.data.mul_(0.0)
        self.actor_mean.weight.data.mul_(0.1)
        self.critic_linear.bias.data.mul_(0.0)
        self.critic_linear.weight.data.mul_(0.1)
        self.actor_log_var = nn.Parameter(torch.zeros(1, num_outputs) -0.7)

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
    return torch.Tensor(obs)


def sample(mu, sigma_sq, gpu):
    x = torch.randn(mu.size())
    x = to_Variable(x, gpu)
    return sigma_sq.sqrt() * x + mu


def normal_pdf(x, mu, sigma_sq):
    a = 1/(2*np.pi*sigma_sq).sqrt()
    b = (-1.*(x-mu).pow(2)/(2*sigma_sq)).exp()
    return a * b 
 

def evaluate(rank, args, shared_model, T, num_steps=250000): 
    # evaluate the shared model, one minute once
    if args.gpu:
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)
    env = gym.make(args.env_name)
    local_model = Net(args.num_inputs, args.num_outputs)
    if args.gpu:
        local_model.cuda()
    total = 0.
    episodes = 0
    obs = env.reset()
    save_freq = 20
    length = 0
    done = False
    while True: 
        state = process_img(obs).unsqueeze(0) 
        state = Variable(state, volatile=True)
        if args.gpu: 
            state = state.cuda()
        mu, sigma_sq = local_model.act(state)
        action = to_numpy(sample(mu, sigma_sq, args.gpu), args.gpu)
        obs, reward, done, info = env.step(action[0])
        total += reward
        length += 1
        if length > args.max_episode_length:
            done = True

        if done:
            obs = env.reset()
            if episodes % save_freq == 0:
                print (time.strftime('%Hh %Mm %Ss', time.localtime()), 'steps ', T.value, 'score ', total, file=open(args.results_file, 'a'))
                torch.save(local_model.state_dict(), args.weights_file+str(episodes//save_freq))
            if best < total:
                torch.save(local_model.state_dict(), args.weights_file+'-best')
                best = total
            length, total = 0, 0
            episodes += 1
            local_model.load_state_dict(shared_model.state_dict())

    return total 

def share_grad(local_model, shared_model):
    # share gradient with the shared model
    for p, shared_p in zip(local_model.parameters(), shared_model.parameters()):
        if shared_p.grad is not None:
            return
        shared_p._grad = p.grad
 


def train(rank, args, shared_model, T):
    if args.gpu:
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)
    gamma = args.gamma
    env = gym.make(args.env_name)
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    local_model = Net(args.num_inputs, args.num_outputs)
    local_model.train()
    if args.gpu:
        local_model.cuda()
    obs = env.reset()
    length = 0
    while True:
        length += 1
        # Sync with the shared model
        local_model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        
        for step in range(args.num_steps):
            state = process_img(obs).unsqueeze(0) 
            if args.gpu:
                state = state.cuda()
            state = Variable(state)
            mu, sigma_sq = local_model.act(state)
            action = to_numpy(sample(mu, sigma_sq, args.gpu), args.gpu)
            prob = normal_pdf(action, mu, sigma_sq)
            log_prob = prob.log()

            obs, reward, done, _ = env.step(to_numpy(action))
            value = local_model.critic(state)


            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                length = 0
                obs = env.reset()
                break

        R = torch.zeros(1, 1)
        if args.gpu:
            R = R.cuda()
        if not done:
            state = process_img(obs).unsqueeze(0) 
            if args.gpu:
                state = state.cuda()
            state = Variable(state)
            value = local_model.critic(state)
            R = value.data

        R = Variable(R)
        values.append(R)
        policy_loss = 0
        value_loss = 0
        #gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            #delta_t = rewards[i] + gamma * \
            #    values[i + 1].data - values[i].data
            #gae = gae * gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(advantage.data) 

        #print value_loss
        #print policy_loss
        optimizer.zero_grad()

        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)

        share_grad(local_model, shared_model)
        optimizer.step()
        with T.get_lock():
            T.value += len(rewards)
                                                                             

parser = argparse.ArgumentParser(description='a3c')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=41, metavar='S', help='random seed (default: 41)')
parser.add_argument('--num-processes', type=int, default=8, metavar='NP', help='number of processes to use')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps')
parser.add_argument('--max-episode-length', type=int, default=200000, metavar='M', help='maximum length of an episode (default: 200000)')
parser.add_argument('--env-name', default='RoboschoolHumanoid-v1', metavar='ENV', help='environment to train on')
parser.add_argument('--no-shared', default=True, metavar='SHR', help='use an optimizer without shared momentum (default: True)')
parser.add_argument('--window', type=int, default=4, metavar='W', help='number of the input frames')
parser.add_argument('--gpu', default=False, metavar='GPU', help='use GPU or not (default: False)')
parser.add_argument('--frame-size', type=int, default=80, metavar='FS', help='size of the input frame')
parser.add_argument('--weights-file', default='a3c-weights', metavar='WF', help='file name for trained weights (default: a3c-weights)')
parser.add_argument('--results-file', default='a3c-results', metavar='RF', help='file name for estimation during training (default: a3c-results)')
parser.add_argument('--tau', type=float, default=0.0001, metavar='T')
parser.add_argument('--num-inputs', type=int, default=44, metavar='NIP', help='number (size) of inputs')
parser.add_argument('--num-outputs', type=int, default=17, metavar='NOP', help='number (size) of outputs')

def main(): 
    #os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    args.gpu = args.gpu and torch.cuda.is_available()
    if args.gpu:
        torch.cuda.manual_seed(45)
        args.weights_file = args.weights_file + '-gpu'
        args.results_file = args.results_file + '-gpu'
    else:
        torch.manual_seed(45)
    args.weights_file = args.env_name + '-' + args.weights_file
    args.results_file = args.env_name + '-' + args.results_file
    
    model = Net(args.num_inputs, args.num_outputs)
    if args.gpu:
        model.cuda()

    print (model)
    model.share_memory()
    T = mp.Value('i', 0)
    process = []
    p = mp.Process(target=evaluate, args=(49, args, model, T))
    p.start()
    process.append(p)
    for i in range(args.num_processes):
        p = mp.Process(target=train, args=(i, args, model, T))
        p.start()
        process.append(p)
    for p in process: 
        p.join()

if __name__ == '__main__':
    main()


