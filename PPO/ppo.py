# computation is done in one process
# use separated actor and critic

import gym, roboschool
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


class Memory(object):
    def __init__(self, size):
        self.mem = deque(maxlen=size)

    def append(self, replay):
        # replay includes state(np.array), action(np.array), reward(np.array), new_state(np.array), done(np.array)
        self.mem.append(replay)

    def __len__(self):
        return len(self.mem)

    def clear(self):
        self.mem.clear()

    def sample(self, size):
        # return tensors
        xp = random.sample(self.mem, min(len(self.mem), size))
        states, actions, values, adv = [], [], [], []
        for i in range(len(xp)):
            s, a, v, ad = xp[i]
            states.append(to_tensor(s))
            actions.append(to_tensor(a))
            values.append(to_tensor(v))
            adv.append(to_tensor(ad))
        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        values = torch.cat(values, 0)
        adv = torch.cat(adv, 0)
        return states, actions, values, adv 





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
    
    

def play(seed, args, shared_model, q, training, T, flag, num_steps):
    env = gym.make(args.env_name)
    obs = env.reset()
    episodes = 0 
    total = 0.
    best = 0.
    save_freq = 5
    local_model = Net(args.num_inputs, args.num_outputs)
    if args.gpu:
        local_model.cuda()
    length = 0 
    steps = 0 
    for _ in range(args.num_iterations):
        local_model.load_state_dict(shared_model.state_dict())
        states, actions, rewards, terminal, adv, values = [], [], [], [], [], []
        for __ in range(num_steps):
            #if False == training: env.render()
            state = process_obs(obs).unsqueeze(0)
            states.append(state.numpy())
            state = Variable(state, volatile=True)
            if args.gpu:
                state = state.cuda()
            mu, sigma_sq = local_model.act(state)
            values.append(to_numpy(local_model.critic(state), args.gpu))
            action = sample(mu, sigma_sq, args.gpu)
            action = to_numpy(action, args.gpu)
            actions.append(action)
            obs, reward, done, info = env.step(action[0])
            total += reward
            length += 1
            steps += 1

            rewards.append(np.array([[float(reward)]]))

            if length > args.max_episode_length:
                done = True

            x = 0. if done else 1.
            terminal.append(np.array([[x]]))

            if done:
                obs = env.reset()
                episodes += 1
                if False == training:
                    if episodes % save_freq == 0:
                        print (time.strftime('%Hh %Mm %Ss', time.localtime()), 'steps ', T.value, 'score ', total, file=open(args.results_file, 'a'))
                        torch.save(local_model.state_dict(), args.weights_file+str(episodes//save_freq))
                    if best < total:
                        torch.save(local_model.state_dict(), args.weights_file+'-best')
                        best = total
                length, total = 0, 0

        # compute advantage function
        values_target = []
        R = torch.zeros(1, 1)
        if args.gpu:
            R = R.cuda()
        if False == done:
            s = process_obs(obs)
            if args.gpu: s = s.cuda()
            value = local_model.critic(Variable(s.unsqueeze(0), volatile=True))
            R = value.data
        if args.gpu:
            R = R.cpu()
        R = R.numpy()
        # apply GAE 
        values.append(R)
        gae = 0
        for i in range(len(rewards))[::-1]:
            R = args.gamma * R * terminal[i] + rewards[i]
            delta = rewards[i] + args.gamma * values[i+1] * terminal[i] - values[i]
            values_target.append(R)
            #adv_est = R - values[i] # advantage estimator
            gae = delta + gae * args.gamma * args.lambd 
            adv.append(gae)
        values_target = values_target[::-1]
        adv = adv[::-1]

        # send data
        for i in range(len(rewards)):
            if q.full():
                time.sleep(0.1)
            q.put((states[i], actions[i], values_target[i], adv[i]))
        T.value += len(rewards)

        flag[seed] = True
        while flag[seed]:
            #time.sleep(0.1)
            pass
    return

def normal_pdf(x, mu, sigma_sq):
    a = 1/(2*np.pi*sigma_sq).sqrt()
    b = (-1.*(x-mu).pow(2)/(2*sigma_sq)).exp()
    return a * b 

def train(args):
    # manage running processes
    q = mp.Queue(1000)
    model = Net(args.num_inputs, args.num_outputs)
    print (model, file=open(args.results_file, 'a'))
    #model.load_state_dict(torch.load('weights16_addpg'))
    model_old = Net(args.num_inputs, args.num_outputs)
    if args.gpu:
        model.cuda()
        model_old.cuda()
    model.share_memory()
    T = mp.Value('i', 0) # count for total number of frames 
    flags = mp.Array('b', args.num_processes+4) # flags for synchronization
    for i in range(len(flags)):
        flags[i] = False
    batch_size = 4096
    gamma = args.gamma
    mem = Memory(100000)
    num_steps = args.num_steps
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #schd = scheduler.StepLR(optimizer, step_size=1200000, gamma=0.9)

    process = []
    p = mp.Process(target=play, args=(0, args, model, q, False, T, flags, num_steps))
    p.daemon=True
    p.start()
    process.append(p)
    for i in range(args.num_processes):
        p = mp.Process(target=play, args=(i+1, args, model, q, True, T, flags, num_steps))
        p.daemon=True
        p.start()
        process.append(p)
    for _ in range(args.num_iterations):
        # collect data
        while sum(flags) != args.num_processes + 1 or False == q.empty():
            mem.append(q.get())

        # training starts
        model_old.load_state_dict(model.state_dict())
        for __ in range(args.num_epochs):
            b_s, b_a, b_v, b_ad = mem.sample(batch_size)
            b_s, b_a, b_v, b_ad = to_Variable(b_s, args.gpu), to_Variable(b_a, args.gpu), to_Variable(b_v, args.gpu), to_Variable(b_ad, args.gpu)

            v = model.critic(b_s)
            l2 = nn.MSELoss()
            l_vf = l2(v, b_v)

            mu, sigma_sq = model.act(b_s)
            probs = normal_pdf(b_a, mu, sigma_sq)
            mu_old, sigma_sq_old = model_old.act(b_s)
            probs_old = normal_pdf(b_a, mu_old, sigma_sq_old)
            ratio = probs / (probs_old + 1e-15)
            l1 = ratio * b_ad
            l2 = ratio.clamp(1 - args.eps, 1 + args.eps) * b_ad
            l_clip = torch.mean(torch.min(l1, l2))
            # no entropy loss for roboschool problems
            loss = l_vf - l_clip
            #print (loss)

            optimizer.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_norm(model.parameters(), 40) 

            optimizer.step()
            #schd.step()

        mem.clear()
        model.actor_log_var.data -= 1e-3
        for j in range(len(flags)):
            flags[j] = False



parser = argparse.ArgumentParser(description='ppo')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=41, metavar='S', help='random seed (default: 41)')
parser.add_argument('--num-processes', type=int, default=8, metavar='NP', help='number of processes to use (default: 1)')
parser.add_argument('--num-steps', type=int, default=2048, metavar='NS', help='number of forward steps (default: 10)')
parser.add_argument('--max-episode-length', type=int, default=20000, metavar='M', help='maximum length of an episode')
parser.add_argument('--env-name', default='RoboschoolHumanoid-v1', metavar='ENV', help='environment to train on')
parser.add_argument('--no-shared', default=True, metavar='SHR', help='use an optimizer without shared momentum (default: True)')
parser.add_argument('--window', type=int, default=4, metavar='W', help='number of the input frames')
parser.add_argument('--gpu', default=True, metavar='GPU', help='use GPU or not (default: False)')
parser.add_argument('--frame-size', type=int, default=80, metavar='FS', help='size of the input frame')
parser.add_argument('--weights-file', default='ppo-weights', metavar='WF', help='file name for trained weights')
parser.add_argument('--results-file', default='ppo-results', metavar='RF', help='file name for estimation during training')
parser.add_argument('--tau', type=float, default=0.0001, metavar='T')
parser.add_argument('--num-inputs', type=int, default=44, metavar='NIP', help='number (size) of inputs')
parser.add_argument('--num-outputs', type=int, default=17, metavar='NOP', help='number (size) of outputs')
parser.add_argument('--num-iterations', type=int, default=1000000, metavar='NIT', help='number of iterations to run')
parser.add_argument('--eps', type=float, default=0.2, metavar='EPS', help='parameter for clipping the loss')
parser.add_argument('--lambd', type=float, default=0.95, metavar='LAM', help='Generalized Advantage Estimator (GAE) parameter')
parser.add_argument('--num-epochs', type=int, default=20, metavar='NE', help='number of epochs for training')


def main():
    #os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    args.gpu = args.gpu and torch.cuda.is_available()
    gpu = args.gpu

    if args.gpu:
        torch.cuda.manual_seed(45)
        args.weights_file = args.weights_file + '-gpu'
        args.results_file = args.results_file + '-gpu'
    else:
        torch.manual_seed(45)
    args.weights_file = args.env_name + '-' + args.weights_file
    args.results_file = args.env_name + '-' + args.results_file
    train(args)

if __name__ == '__main__':

    main()


                      

