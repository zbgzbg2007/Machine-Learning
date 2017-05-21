"""Run Atari Environment with Agent.
   Shared network version.  
"""
import random
import gym
import multiprocessing
import keras
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input, Permute) 
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from scipy import ndimage

class MyModel(object):
    ''' The policy network model. 
    '''


    def create_model(self, input_shape, num_actions):  
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

        inputs = Input(shape=input_shape)
        action_inputs = Input(shape=(9,))
        reward_inputs = Input(shape=(1,))
        x = Permute((2, 3, 1))(inputs)
    
        x = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)

        # two outputs of the network: value function approximator and probability for actions
        val = Dense(units=1, activation='linear', name='val_output')(x)
        p = Dense(units=num_actions, activation='softmax', name='pi_output')(x)

        # the two outputs are concatenated 
        outputs = keras.layers.concatenate([val, p])
        model = Model(inputs=[inputs, action_inputs, reward_inputs], outputs=outputs)

        return model

    def __init__(self, in_dim, num_actions, fname):
        self.network = self.create_model(in_dim, num_actions)
        self.num_actions = num_actions
        self.fname = fname
        self.compile()


    def loss(self, y_true, y_pred):
        ''' Loss function. 
  
        - Input
          - y_true: tf.Tensor
            Target value
          - y_pred: tf.Tensor
            Predicted value          
        
        - Output
          - loss_total: tf.Tensor
            Total loss.
        '''
        # obtain state value v and action probability pi
        v = tf.slice(y_pred, [0, 0], [-1, 1])
        pi = tf.slice(y_pred, [0, 1], [-1, -1])
 
        tf.squeeze(v, [1])
        log_pi = tf.log(pi + 1e-10)
        log_pi_s = tf.reduce_sum(log_pi*self.network.inputs[1], axis=1)
        # note the stop_gradient function
        adv = tf.subtract(tf.stop_gradient(v), self.network.inputs[2])
       
        # the loss for the policy part
        loss_pi = tf.reduce_sum(log_pi_s * adv)

        # the loss for the state value part
        loss_v = tf.nn.l2_loss(v-self.network.inputs[2])

        # the loss for the entropy of the policy
        loss_entropy = tf.reduce_sum(pi*log_pi)
        entropy_beta = tf.constant(0.01)

        loss_total = tf.add_n([loss_pi, loss_v, loss_entropy*entropy_beta])
        loss_total = tf.truediv(loss_total, tf.cast(tf.shape(self.network.inputs[2])[0], tf.float32))
 
        return loss_total

    def compile(self):
        ''' Compile the network.
        '''
        opt = keras.optimizers.Adam(lr=0.0002)
        self.network.compile(optimizer=opt, loss=self.loss)

    def predict(self, state, batch_size):
        return self.network.predict(state, batch_size=batch_size)

    def fit(self, act, rewards, x, y, batch_size, verbose=0):
        self.network.fit([x, act, rewards], y, batch_size=batch_size, verbose=verbose)
            
    def save_weights(self): 
        self.network.save_weights(self.fname)

def process_state(obs):
        """Preprocess the given state before giving it to the model.
           Convert the image to greyscale and downsample the image.  

        - Input
          - obs: np.ndarray
            A single observation from an environment.

        - Output
          - processed_state: np.ndarray
            The state after processing. Can be modified in anyway.

        """
        #state = obs[:160, ...] 
        # convert RGB image into grayscale
        state = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        # resize image into 80 * 80 
        state = cv2.resize(state[:160, :], (80, 80)) / 255.0
        
        return state

def run(game, conn, T_max, jump, window):
    '''Run a game for T_max steps

    - Input
      - game: string
        Game name to run
      - conn: connection object
        End of a pipe between this processor and the main processor
      - T_max: int
        The max steps to run
      - jump: int
        The number of random actions are done at the beginning of the run.
        This is only for the first episode.
      - window: int
        The number of frames as input for the network

    - Output
    '''
    env = gym.make(game)
    history = np.empty((window, 80, 80))
    T = 0
    t_max = 5
    flag  = False
    terminal = True
    while T < T_max+jump: 
        if T >= jump-1: 
            flag = True        
        if terminal:
            # if it is the start of a new episode, choose random action
            terminal = False
            obs = env.reset()
            for i in range(window):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                s = process_state(obs)
                history[i] = s
            index = 0
    
        memory = [] # store the last t_max transformations
        for _ in range(t_max):    
            # compute the next t_max states by the same action
            T += 1
            if flag == False:
                action = env.action_space.sample()
            else:
                conn.send(history[None, ...])
                action = conn.recv()
                     
            obs, r, done, info = env.step(action)
            r = np.clip(r, -1., 1.)
            if done:
                terminal = True
                if _ != t_max-1:
                    conn.send(np.empty(0))
                    conn.recv()
                break

            if flag == True:
                memory.append((history[None, ...], action, r))
            frame = process_state(obs)
            history[index%window] = frame
            index += 1
            
        if flag == False: continue
		
        if terminal:
            # if the last state is terminal
            conn.send((False, 0, memory))
        else:
            # else, compute next state value
            conn.send((True, history[None, ...], memory))


def evaluate(model, env, eps, num_episodes, window, max_episode_length=1000000):
        """Test the agent with a provided environment.
        
        - Input
          - model: MyModel
            The network model 
          - env: gym.Env
            This is the Atari environment. 
          - eps: float
            The epsilon for the eps-greedy policy
          - num_episodes: int
            How many samples to perform.
          - window: int
            The number of the frames as input
          - max_episode_length: int
            How long a single episode should last before the agent

        - Output
          - total: float
            the cumulative rewards from all episodes
        """
        total = 0.0 
        num_actions = env.action_space.n
        history = np.empty((window, 80, 80)) 
        for episode in range(num_episodes):
            obs = env.reset()
            index = 0

            for i in range(window):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action) 
                s = process_state(obs) 
                history[i] = s
            for t in range(max_episode_length):
                y = model.predict([history[None, ...], np.empty(9)[None,...], np.empty(1)[None, ...]], batch_size=1)
                pi = y[0, 1:]
                if random.random() <= eps:
                   action = random.randint(0, num_actions-1)
                else:
                   action = np.random.choice(num_actions, p=pi)
                #env.render()
                obs, reward, done, info = env.step(action)
                total += reward
                if done:
                    break
                state = process_state(obs)
                history[index%window] = state
                index += 1

        return total


def main():

    # setup parameters
    game = 'Enduro-v0'
    env = gym.make(game)
    num_actions = env.action_space.n
    window = 4
    in_dim = (window, 80, 80)
    file_name = 'trained_weights'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
    model = MyModel(in_dim, env.action_space.n, file_name)
    #model.network.load_weights(file_name)
    #evaluate(model, env, 0.1, 2, window)

    eps = [1.] * 30
    eps_min = [0.01, 0.2, 0.1, 0.05, 0.4, 0.01, 0.5, 0.1, 0.1, 0.01, 0.15, 0.1, 0.5, 0.1, 0.2, 0.1, 0.1, 0.05, 0.3, 0.1, 0.2, 0.2, 0.15, 0.1, 0.25, 0.05, 0.5, 0.25, 0.15, 0.2]
    jump = [10, 40, 300, 350, 660, 170, 520, 200, 120, 380, 420, 500, 600, 100, 240, 130, 790, 280, 150, 440, 20, 900, 550, 410, 240, 330, 580, 630, 110, 170] # number of steps to run randomly at the beginning
    T_max = 1000000
    t_max = 5
    T = 0 
    eval_freq = 30000
    gamma = 0.99

    # run all processors
    processors = []
    conns = []
    nums = len(eps)#multiprocessing.cpu_count()
    for i in range(nums):
        conn1, conn2 = multiprocessing.Pipe()
        p = multiprocessing.Process(target=run, args=(game, conn2, T_max, jump[i], window, keep, keep_length[i]))
        p.start()
        processors.append(p)
        conns.append(conn1)
        
    while T < T_max:
        s = np.empty(in_dim)[None, ...]
        done = [False for i in range(nums)]
        for _ in range(t_max):
            for i in range(0, nums):
                if done[i] == True: continue
                s = conns[i].recv()
                if s.shape[0] == 0: 
                    done[i] = True 
                    conns[i].send([])
                else:
                    if random.random() <= eps[i]:
                        a = random.randint(0, num_actions-1)
                    else:
                        y = model.predict([s, np.empty(9)[None,...], np.empty(1)[None, ...]], batch_size=1)
                        pi = y[0, 1:]
                        a = np.random.choice(num_actions, p=pi)
                    conns[i].send(a)
                    eps[i] = max(eps_min[i], eps[i] - 5e-6)

            T += 1
        s, a, r = np.empty(in_dim)[None, ...], np.empty(num_actions)[None, ...], np.empty(1)[None, ...]
        for i in range(nums):
            flag, next_s, memory = conns[i].recv()
            if flag == False:
                R = 0
            else:
                y = model.predict([next_s, np.empty(9)[None,...], np.empty(1)[None,...]], batch_size=1)
                R = y[0, 0]
            size = len(memory)
            for i in range(size-1, -1, -1):
                R = memory[i][2] + gamma*R
                act = np.zeros(num_actions, dtype=np.int)
                act[memory[i][1]] = 1
                s = np.vstack((s, memory[i][0]))
                a = np.vstack((a, act[None, ...]))
                r = np.vstack((r, R[None, ...]))
        s = s[1:, ...]
        a = a[1:, ...]
        r = r[1:, ...]
        
        model.fit(a, r, s, np.empty((a.shape[0], num_actions+1)), batch_size=a.shape[0], verbose=0)

        if T % eval_freq == 0:
            print 'step %d, evaluate %f' % (T, evaluate(model, env, 0.1, 2, window))
            if T % (eval_freq * 2) == 0: 
                # save weights
                model.save_weights()
    
    print 'main processor exit'

if __name__ == '__main__':
    main()
