# computation is done in one process
# use ddpg network and replay memory

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

class E_args(object):
    def __init__(self, eps, decay, emin, sigma, t, freq, num, gpu):
        self.eps = eps
        self.decay = decay
        self.eps_min = emin
        self.mu = to_tensor([[0.0 for i in range(num)]])
        self.theta = to_tensor([[t for i in range(num)]])
        self.sigma = to_tensor([[sigma for i in range(num)]])
        self.update_freq = freq
        self.gpu = gpu
        self.num_actions = num
        if gpu:
            self.mu = self.mu.cuda() 
            self.theta = self.theta.cuda()
            self.sigma = self.sigma.cuda()
        


class Memory(object):
    def __init__(self, size):
        self.mem = deque(maxlen=size)

    def append(self, replay):
        # replay includes state(np.array), action(np.array), reward(np.array), new_state(np.array), done(np.array)
        self.mem.append(replay)

    def __len__(self):
        return len(self.mem)

    def sample(self, size):
        # return tensors
        xp = random.sample(self.mem, min(len(self.mem), size))
        states, actions, rewards, new_states, done = [], [], [], [], []
        for i in range(len(xp)):
            s, a, r, n, d = xp[i]
            states.append(to_tensor(s))
            actions.append(to_tensor(a))
            rewards.append(to_tensor(r))
            new_states.append(to_tensor(n))
            done.append(to_tensor(d))
        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        rewards = torch.cat(rewards, 0)
        new_states = torch.cat(new_states, 0)
        done = torch.cat(done, 0)
        return states, actions, rewards, new_states, done
            

       


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        # num_inputs (int): size of observation
        # num_outputs (int): size of action
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(num_inputs+num_outputs, 256)
        self.fc4 = nn.Linear(256, 128)

        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)

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


    def critic(self, inputs, action):
        y = torch.cat((inputs, action), 1)
        y = F.relu(self.fc3(y))
        y = F.relu(self.fc4(y))

        return self.critic_linear(y)



def process_obs(obs):
    # process observation 
    return torch.Tensor(obs)



def OU(x, args):
    r = to_tensor(np.random.randn(1, args.num_actions))
    if args.gpu:
        r = r.cuda() 
    return args.theta * (args.mu - x) + args.sigma * r

def explore(action, eps_args):
    # add noise to action
    noise = to_Variable(OU(action.data, eps_args), eps_args.gpu)
    action = Variable(action.data) + noise 
    return action


def play(seed, args, shared_model, q=None, eps_args=None, training=False, T=None, num_steps=10000000):
    if args.gpu:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    env = gym.make(args.env_name)
    obs = env.reset()
    num_episodes = 0
    total = 0.
    best = 0.
    save_freq = 20
    local_model = Net(args.num_inputs, args.num_outputs)
    local_model.load_state_dict(shared_model.state_dict())
    if args.gpu:
        local_model.cuda()
    states, actions, rewards, terminal = [], [], [], []
    length = 0
    steps = 0
    while steps < num_steps:
        if training and 0 == steps % eps_args.update_freq:
            local_model.load_state_dict(shared_model.state_dict())
        state = process_obs(obs).unsqueeze(0)
        states.append(state.numpy())
        state = Variable(state, volatile=True)
        if args.gpu:
            state = state.cuda() 
        action = local_model.act(state)
        if random.random() < eps_args.eps:
            action = explore(action, eps_args)
        eps_args.eps = max(eps_args.eps - eps_args.decay, eps_args.eps_min)
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
            if False == training:
                num_episodes += 1
                if num_episodes % save_freq == 0:
                    print (time.strftime('%Hh %Mm %Ss', time.localtime()), 'steps ', T.value, 'score ', total, file=open(args.results_file, 'a'))
                    torch.save(local_model.state_dict(), args.weights_file+str(num_episodes//save_freq))
                if best < total:
                    torch.save(local_model.state_dict(), args.weights_file+'-best')
                    best = total
                local_model.load_state_dict(shared_model.state_dict())
            length, total = 0, 0 
        if 0 == steps % args.num_steps:
            for i in range(len(rewards)-1):
                if q.full():
                    time.sleep(0.1)
                if 0 != rewards[i]:
                    q.put((states[i], actions[i], rewards[i], states[i+1], terminal[i]))
            with T.get_lock():
                T.value += len(rewards)-1
            states, actions, rewards, terminal = states[-1:], actions[-1:], rewards[-1:], terminal[-1:]
  
    return 


def soft_update(s, t, tau):
    # update target network t parameter by s*tau + (1-tau)*t
    for sp, tp in zip(s.parameters(), t.parameters()):
        tp.data.copy_(tp.data * (1-tau) + sp.data * tau)

def manager(args):
    # manage running processes
    q = mp.Queue(5000)
    model = Net(args.num_inputs, args.num_outputs)
    print (model, file=open(args.results_file, 'a'))
    #model.load_state_dict(torch.load('weights16_addpg'))
    target_model = Net(args.num_inputs, args.num_outputs)
    target_model.load_state_dict(model.state_dict())
    if args.gpu: 
        model.cuda() 
        target_model.cuda()
    model.share_memory()
    target_model.share_memory()
    T = mp.Value('i', 0) # count for total number of frames 
    
    eps = [1, 0.6, 0.7, 0.8, 0.9, 1., 0.85, 0.4, 0.8, 0.7, 0.1]
    eps_min = [0.0, 0.0, 0.1, 0.15, 0.05, 0.0, 0.15, 0.8, 0.7, 0.1]
    decay = [1e-4, 1e-6, 1e-6, 1e-6, 1e-6, 2e-6, 2e-6, 3e-5, 4e-5, 2e-5]
    sigma = [0.1, 0.15, 0.25, 0.35, 0.45, 0.5, 0.1, 0.2, 0.05, 0.03]
    update_freq = [10, 20, 30, 40, 50, 50, 90, 100]
    theta = [0.65, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6]
    eps_args = []
    for i in range(args.num_processes+1):
        eps_args.append(E_args(eps[i], decay[i], eps_min[i], sigma[i], theta[i],  update_freq[i], args.num_outputs, args.gpu)) 
    num_steps = 300000000
    p = mp.Process(target=train, args=(args, model, target_model, q))
    p.daemon = True
    p.start()
    process = []
    p = mp.Process(target=play, args=(50, args, model, q, eps_args[0], False, T, num_steps))
    p.start()
    process.append(p)
    for i in range(args.num_processes):
        p = mp.Process(target=play, args=(i, args, model, q, eps_args[i+1], True, T, num_steps))
        p.start()
        process.append(p) 
    for p in process:
        p.join()


def train(args, model, target_model, q):
    batch_size = 4096 
    gamma = 0.99
    mem = Memory(1000000)
    actor_optim = optim.Adam([{'params': model.fc1.parameters()}, {'params': model.fc2.parameters()}, {'params': model.actor_linear.parameters()}], lr=args.lr)
    critic_optim = optim.Adam([{'params': model.fc3.parameters()}, {'params': model.fc4.parameters()}, {'params': model.critic_linear.parameters()}], lr=args.lr)
    actor_optim = scheduler.StepLR(actor_optim, step_size=1000000, gamma=0.9)
    critic_optim = scheduler.StepLR(critic_optim, step_size=1000000, gamma=0.9)
    while True: 
        # add more replay
        count = 0
        while False == q.empty() and count < 200:
            mem.append(q.get())
            count += 1
        if (len(mem) >= 5000):
            b_s, b_a, b_r, b_n, b_t = mem.sample(batch_size)
            b_s, b_a, b_r, b_n, b_t = to_Variable(b_s, args.gpu), to_Variable(b_a, args.gpu), to_Variable(b_r, args.gpu), to_Variable(b_n, args.gpu), to_Variable(b_t, args.gpu) 
            next_v = target_model.critic(b_n, target_model.act(b_n))
            next_v = to_Variable(next_v.data, args.gpu)
            target_v = b_r + gamma * next_v * b_t
            v = model.critic(b_s, b_a)
            l2 = nn.MSELoss()
            value_loss = l2(v, target_v) 
            policy_loss = -model.critic(b_s, model.act(b_s)).mean()


            critic_optim.optimizer.zero_grad()
            value_loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 40)

            critic_optim.optimizer.step()
            critic_optim.step()
            
            actor_optim.optimizer.zero_grad()

            policy_loss.backward() 

            torch.nn.utils.clip_grad_norm(model.parameters(), 40)

            actor_optim.optimizer.step()
            actor_optim.step()

            soft_update(model, target_model, args.tau)

        else:
            pass
        
       
  
                                                                             

parser = argparse.ArgumentParser(description='bddpg')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=41, metavar='S', help='random seed (default: 41)')
parser.add_argument('--num-processes', type=int, default=6, metavar='NP', help='number of processes to use (default: 1)')
parser.add_argument('--num-steps', type=int, default=10, metavar='NS', help='number of forward steps (default: 10)')
parser.add_argument('--max-episode-length', type=int, default=20000, metavar='M', help='maximum length of an episode')
parser.add_argument('--env-name', default='RoboschoolHumanoid-v1', metavar='ENV', help='environment to train on')
parser.add_argument('--no-shared', default=True, metavar='SHR', help='use an optimizer without shared momentum (default: True)')
parser.add_argument('--gpu', default=True, metavar='GPU', help='use GPU or not (default: False)')
parser.add_argument('--weights-file', default='bddpg-weights', metavar='WF', help='file name for trained weights')
parser.add_argument('--results-file', default='bddpg-results', metavar='RF', help='file name for estimation during training')
parser.add_argument('--tau', type=float, default=0.0001, metavar='T', help='parameter for soft updating target network')
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
    manager(args)

if __name__ == '__main__':

    main()


