import gym
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
gpu = False


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if gpu:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def to_numpy(x):
    # return numpy of variable
    return x.cpu().data.numpy() if gpu else x.data.numpy()


def init_weight(layer, nonlinear):
    # initialize weights
    init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain(nonlinear))
    init.constant(layer.bias, 0.) 

class Net(nn.Module):
    def __init__(self, num_outputs, num_inputs=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc = nn.Linear(2048, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        init_weight(self.conv1, 'relu')
        init_weight(self.conv2, 'relu')
        init_weight(self.fc, 'relu')
        init_weight(self.critic_linear, 'linear')
        init_weight(self.actor_linear, 'linear')

        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.critic_linear(x), self.actor_linear(x)



def process_img(obs):
    # process image
    obs = scipy.misc.imresize(obs, (80, 80))
    obs = obs.astype(np.float32)
    obs = obs / 255.
    return torch.Tensor(np.transpose(obs, (2, 0, 1)))


def evaluate(rank, args, shared_model, num_steps=250000): 
    # evaluate the shared model, one minute once
    if args.gpu:
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)
    env = gym.make(args.env_name)
    local_model = Net(env.action_space.n)
    local_model.eval()
    if args.gpu:
        local_model.cuda()
    total = 0.
    num_actions = env.action_space.n
    num_episodes = 0
    actions = deque(maxlen=400)
    obs = env.reset()
    save_freq = 30
    start_time = time.time()
    episode_length = 0
    state = process_img(obs)
    done = False
    while True: 
        episode_length += 1
        if done:
            obs = env.reset()
            state = process_img(obs)
            print ('Time {}, episode reward {}, episode length {}'.format(time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time)), total, episode_length))
            episode_length = 0
            num_episodes += 1
            print (total, file=open(args.results_file, 'a'))
            total = 0.
            actions.clear()
            #if num_episodes % save_freq == 0:
                #torch.save(local_model.state_dict(), g_fname[:])
                #print '%d episodes, weights saved' % num_episodes
            time.sleep(60)
            local_model.load_state_dict(shared_model.state_dict())
        state = state.unsqueeze(0)
        if args.gpu: 
            state = state.cuda()
        state = Variable(state)
        _, pi = local_model(state)
        pi = F.softmax(pi)
        action = to_numpy(pi.max(1)[1])
        #action = pi.multinomial().data
        obs, reward, done, info = env.step(action[0, 0])
        state = process_img(obs)
        total += reward
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

    return total 

def share_grad(local_model, shared_model):
    # share gradient with the shared model
    for p, shared_p in zip(local_model.parameters(), shared_model.parameters()):
        if shared_p.grad is not None:
            return
        shared_p._grad = p.grad
 


def train(rank, args, shared_model, eps):
    if args.gpu:
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)
    gamma = 0.99
    env = gym.make(args.env_name)
    optimizer = optim.Adam(shared_model.parameters(), lr=0.0001)
    local_model = Net(env.action_space.n)
    local_model.train()
    if args.gpu:
        local_model.cuda()
    obs = env.reset()
    state = process_img(obs)
    decay = 5e-4
    eps_min = .1
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        local_model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(args.num_steps):
            state = state.unsqueeze(0)
            if args.gpu:
                state = state.cuda()
            state = Variable(state)
            value, logit = local_model(state)
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)
           
            action = prob.multinomial() if random.random() <= eps else prob.max(1)[1]
            eps = min(eps - decay, eps_min)
            log_prob = log_prob.gather(1, Variable(action.data))

            obs, reward, done, _ = env.step(to_numpy(action))
            #print ('train pi', prob)
            #print ('train value', value)
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                obs = env.reset()

            state = process_img(obs)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if args.gpu:
            R = R.cuda()
        if not done:
            if args.gpu:
                state = state.cuda()  
            value, _ = local_model((Variable(state.unsqueeze(0))))
            R = value.data

        R = Variable(R)
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            #delta_t = rewards[i] + gamma * \
            #    values[i + 1].data - values[i].data
            #gae = gae * gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(advantage.data) - 0.01 * entropies[i]

        #print value_loss
        #print policy_loss
        optimizer.zero_grad()

        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm(local_model.parameters(), 40)

        share_grad(local_model, shared_model)
        optimizer.step()
                                                                             

parser = argparse.ArgumentParser(description='a3c')
parser.add_argument('--lr', type=float, default=0.00014, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=41, metavar='S', help='random seed (default: 41)')
parser.add_argument('--num-processes', type=int, default=16, metavar='NP', help='number of processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5, metavar='NS', help='number of forward steps (default: 5)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M', help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='Pong-v0', metavar='ENV', help='environment to train on (default: Pong-v0)')
parser.add_argument('--no-shared', default=True, metavar='SHR', help='use an optimizer without shared momentum (default: True)')
parser.add_argument('--window', type=int, default=4, metavar='W', help='number of the input frames')
parser.add_argument('--gpu', default=False, metavar='GPU', help='use GPU or not (default: False)')
parser.add_argument('--frame-size', type=int, default=80, metavar='FS', help='size of the input frame')
parser.add_argument('--weights-file', default='a3c-weights', metavar='WF', help='file name for trained weights (default: a3c-weights)')
parser.add_argument('--results-file', default='a3c-results', metavar='RF', help='file name for estimation during training (default: a3c-results)')
parser.add_argument('--tau', type=float, default=1.0, metavar='T')

def main(): 
    os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    args.gpu = args.gpu and torch.cuda.is_available()
    gpu = args.gpu
    if args.gpu:
        torch.cuda.manual_seed(45)
        args.weights_file = args.weights_file + '-gpu'
        args.results_file = args.results_file + '-gpu'
    else:
        torch.manual_seed(45)
    args.weights_file = args.env_name + args.weights_file
    args.results_file = args.env_name + '-' + args.results_file
    env = gym.make(args.env_name)
    model = Net(env.action_space.n)

    print (model)
    model.share_memory()
    if args.gpu:
        model.cuda()
    process = []
    p = mp.Process(target=evaluate, args=(49, args, model,))
    p.start()
    process.append(p)
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for i in range(args.num_processes):
        p = mp.Process(target=train, args=(i, args, model, eps[i]))
        p.start()
        process.append(p)
    for p in process: 
        p.join()
    torch.save(model.state_dict(), args.weights_file)

if __name__ == '__main__':
    main()

