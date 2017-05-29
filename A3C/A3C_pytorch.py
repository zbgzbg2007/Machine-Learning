import gym
import torch
import random
import scipy.misc
from multiprocessing import Value, Array
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as T
import torch.autograd as autograd

g_USE_CUDA = Value('b', False)
g_game = Array('c', 'Breakout-v0')
g_fname = Array('c', 'Breakout-v0-weights')
g_fresults = Array('c', 'Breakout-v0-outputs')
g_window = Value('i', 4)
g_in_dim = Array('i', [4, 100, 100])
g_eps_min = Array('d', [0.1, 0.25, 0.05, 0.5, 0.1, 0.05, 0.5, 0.1, 0,1, 0.3, 0.2, 0.15, 0.25, 0.1, 0.5, 0.05])
g_T = Value('i', 50000000)
g_t = Value('i', 0)
g_gamma = Value('d', 0.99)
g_decay = Value('d', 4e-6)
g_evl = Value('i', 100000)

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if g_USE_CUDA.value:
            data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

def init_weight(layer, nonlinear):
    init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain(nonlinear))
    init.constant(layer.bias, 0.) 


class Net(nn.Module):
    def __init__(self, H, W, num_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(g_window.value, 32, kernel_size=8, stride=4)
        init_weight(self.conv1, 'relu')
        H = (H - 8)/4 + 1
        W = (W - 8)/4 + 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        init_weight(self.conv2, 'relu')
        H = (H - 4)/2 + 1
        W = (W - 4)/2 + 1
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        init_weight(self.conv3, 'relu')
        H = (H - 3)/1 + 1
        W = (W - 3)/1 + 1
        self.fc1 = nn.Linear(H*W*64, 512)
        init_weight(self.fc1, 'relu')
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        init.constant(self.fc1.bias, 0.) 
        self.fc2 = nn.Linear(512, 1)
        init_weight(self.fc2, 'linear')
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        init.constant(self.fc2.bias, 0.) 
        self.fc3 = nn.Linear(512, num_actions)
        init_weight(self.fc3, 'linear')
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        init.constant(self.fc3.bias, 0.) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        v = self.fc2(x)
        pi = self.fc3(x)
        return v, pi

def process_img(obs, last_obs, size):
    obs = np.maximum(obs, last_obs)
    obs = scipy.misc.imresize(obs, size)
    obs = np.dot(obs, [0.299, 0.587, 0.114]) / 255.
    return torch.Tensor(obs[None, ...])

def optimize(model, states, rewards, actions, optm):
    states, rewards, actions = Variable(states), Variable(rewards), Variable(actions)
    v, p = model(states)
    softmax = nn.Softmax()
    pi = softmax(p)
    log_softmax = nn.LogSoftmax()
    log_pi = log_softmax(p)
    log_pi_s = log_pi.gather(1, actions)
    adv = v - rewards
    # requires_grad of this Variable is False
    non_grad_adv = Variable(adv.data)
    loss_pi = torch.sum(log_pi_s * non_grad_adv)
    loss_v = torch.sum(0.5 * adv.pow(2))
    entropy_beta = Variable(torch.Tensor([0.01]))
    loss_entropy = torch.mean(log_pi * pi) * entropy_beta
    loss_total = loss_pi + loss_v - loss_entropy 
    #print 'loss_pi ', loss_pi
    #print 'loss_v ', loss_v
    #print 'loss_entropy ', loss_entropy

    optm.zero_grad()
    loss_total.backward()
    optm.step()
    



def select_action(model, state, eps, num_actions):
    if random.random() > eps:
        _, pi = model(Variable(state))
        softmax = nn.Softmax()
        pi = softmax(pi).data[0].numpy()
        #print pi
        a = np.random.choice(num_actions, p=pi)
    else:
        a = random.randint(0, num_actions-1)
    return torch.LongTensor([[a]])


def evaluate(model, num_episodes, eps=0.1, num_steps=250000):
    env = gym.make(g_game[:])
    total = 0.
    in_dim = tuple(g_in_dim[:])
    num_actions = env.action_space.n
    for _ in range(num_episodes):
        obs = env.reset()
        history = []
        for __ in range(g_window.value):
            action = env.action_space.sample()
            last_obs = obs
            obs, reward, done, info = env.step(action)
            history.append(process_img(obs, last_obs, (in_dim[1], in_dim[2])))
        for i in range(num_steps):
            state = torch.cat(history).unsqueeze(0)
            action = select_action(model, state, eps, num_actions)
            last_obs = obs
            obs, reward, done, info = env.step(action[0, 0])
            total += reward
            if done: break
            history = history[:-1] + [process_img(obs, last_obs, (in_dim[1], in_dim[2]))]
    return total 

def train(model, eps_min, eps = 1.):
    USE_CUDA = g_USE_CUDA.value 
    window = g_window.value
    in_dim = tuple(g_in_dim[:])
    T = g_T.value
    gamma = g_gamma.value 
    decay = g_decay.value 
    env = gym.make(g_game[:])
    t_max = 5
    num_actions = env.action_space.n
    done = True
    num_actions = env.action_space.n
    optm = optim.RMSprop(model.parameters(), lr=0.0002)
    while g_t.value < T:
        if done: 
            done = False
            obs = env.reset()
            history = []
            for _ in range(window):
                action = env.action_space.sample()
                last_obs = obs
                obs, reward, done, info = env.step(action)
                history.append(process_img(obs, last_obs, (in_dim[1], in_dim[2])))
        
        states, rewards, actions = [], [], []
        for i in range(t_max):
            g_t.value += 1
            if (g_t.value+1) % 1 == 0:
                g_t.value += 1
                total = evaluate(model, 4) / 4.
                print 'Step %d Evaluate: %f' % (g_t.value, total)
                with open(g_fresults[:], 'a') as results:
                    print >> results, total
                torch.save(model.state_dict(), g_fname[:])
                g_t.value -= 1
            state = torch.cat(history).unsqueeze(0)
            action = select_action(model, state, eps, num_actions)
            eps = max(eps - decay, eps_min)
            last_obs = obs
            obs, reward, done, info = env.step(action[0, 0])
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            history = history[:-1] + [process_img(obs, last_obs, (in_dim[1], in_dim[2]))]
            if done: break
        if done:
            R = 0.
        else: 
            v, _ = model(Variable(torch.cat(history).unsqueeze(0)))
            R = v.data[0, 0]
        r = []
        for i in range(len(rewards)-1, -1, -1):
            R = gamma*R + rewards[i]
            r.append(torch.Tensor([[R]]))
        states, rewards, actions = torch.cat(states), torch.cat(r), torch.cat(actions)
        optimize(model, states, rewards, actions, optm)

def main(): 
    env = gym.make(g_game[:])
    model = Net(g_in_dim[1], g_in_dim[2], env.action_space.n)
    #model.loat_state_dict(torch.load(g_fname[:]))
    #evaluate(4) / 4.
    #print model
    model.share_memory()
    if g_USE_CUDA.value:
        model.cuda()
    nums = len(g_eps_min)
    process = []
    for i in range(nums):
        p = mp.Process(target=train, args=(model, g_eps_min[i],))
        p.start()
        process.append(p)
    for p in process: 
        p.join()

if __name__ == '__main__':
    main()

