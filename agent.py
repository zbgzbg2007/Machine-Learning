"""Main agent class"""
import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
import keras
import random
from scipy import ndimage
class Agent:
    """Class implementing the agent.

    - Parameters
      - network: keras.models.Model
        The policy network model.
      - in_dim: tuple
        The input shape for the network
      - num_actions: int
        The number of possible actions 
      - epsilon: float
        The constant used in the greedy epsilon policy
      - eps_min: float
        The minimum epsilon after decay
      - gamma: float
        Discount factor for the reward values.
      - global_weights: a list of numpy array
        The global weights for the network.
      - T: int
        The global count for the number of training steps of all threads
      - T_max: int
        The global number representing the total number of training steps needed
      - t_max: int
        The number of game steps for each training step
      - learning_rate: float
        The learning rate used in the network training
      - actions: numpy array of shape (None, num_actions)
        The array representing the chosen actions for the inputs
      - rewards: numpy array of shape (None, 1)
        The array representing rewards for the inputs
    """
    def __init__(self,
                 network,
                 in_dim,
                 num_actions,
                 eps,
                 eps_min,
                 gamma,
                 global_w,
                 T,
                 T_max,
                 t_max=5,
                 learning_rate=0.001):
        self.network = network
        self.in_dim = in_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.epsilon = eps
        self.eps_min = eps_min
        self.gamma = gamma
        self.global_weights = global_w
        self.T = T
        self.T_max = T_max
        self.t_max = t_max
        self.actions = np.zeros(self.num_actions)
        self.rewards = np.zeros(1)
        


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
        v, pi = y_pred[:, 0], y_pred[:, 1:]
 
        log_pi = tf.log(tf.reduce_sum(pi*self.actions, axis=1, keep_dims=True) + 1e-10)
        adv = self.rewards - v
       
        # the loss for the policy part
        # note the minus and the stop_gradient function
        loss_pi = -log_pi * tf.stop_gradient(adv)

        # the loss for the state value part
        loss_v = 0.5 * tf.square(adv)

        # the loss for the entropy of the policy
        loss_entropy = 0.01 * tf.reduce_sum(pi*tf.log(pi + 1e-10), axis=1, keep_dims=True)

        loss_total = tf.reduce_mean(loss_pi + loss_v + loss_entropy)
 
        return loss_total 

    def compile(self):
        ''' Compile the network.
        '''
        opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.99, decay=1e-5)        
        self.network.compile(optimizer=opt, loss=self.loss)

        # compile the predict and train function before applied in multiple threads
        self.network._make_predict_function()
        self.network._make_train_function()
    
    def process_state(self, state):
        """Preprocess the given state before giving it to the network.
           Convert the image to greyscale and downsample the image.  

        - Input
          - state: np.ndarray
            A single observation from an environment.

        - Output
          - processed_state: np.ndarray
            The state after processing. Can be modified in anyway.

        """
        # convert RGB image into grayscale
        gray = np.dot(state, [0.299, 0.587, 0.114])
        
        # downsample image by averaging 2x2 pixels into a single pixel
        fact = 2
        sx, sy = gray.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        regions = sy/fact * (X/fact) + Y/fact
        res = ndimage.mean(gray, labels=regions, index=np.arange(regions.max() + 1))
        res = res.reshape(sx/fact, sy/fact)
        return res


    def select_action(self, state):
        """Select the action based on state by greedy epsilon policy.
           Decrease epsilon if possible.

        - Input
          - state: numpy array
            The input for the network
        
        - Output
          - v: numpy array of shape (1,)
            The state value for the given state
          - a: int
            The chosen action for the given state
        """
        
        y = self.network.predict(state, batch_size=1)
        v, pi = y[:, 0], y[:, 1:]
        if random.random() <= self.epsilon:
           a = random.randint(0, self.num_actions-1)
        else:
           a = np.random.choice(self.num_actions, p=pi[0])
        self.epsilon = max(self.eps_min, self.epsilon - 1e-6)
        
        return v, a


    def fit(self, env):
        """Fit the model to the provided environment.  

        - Input
          - env: gym.Env
            This is the Atari environment. 
        """
        history = []  # store the last four frames

        while self.T < self.T_max: 
            if len(history) == 0: 
                # if it is the start of a new episode, choose random action
                obs = env.reset()
                s = self.process_state(obs)
                s = s[:, :, None]
                history = [s]
                action = env.action_space.sample()

                for _ in range(3):
                    obs, reward, done, info = env.step(action)
                    s = self.process_state(obs)
                    s = s[:, :, None]
                    history.append(s) 
                state = np.vstack((history[0], history[1], history[2], history[3]))    
                v, action = self.select_action(state[None, :])
                
            memory = [] # store the last t_max transformations
            for _ in range(self.t_max):            
                # compute the next t_max states by the same action
                obs, reward, done, info = env.step(action)
                self.T += 1
                if done:
                    history = []
                    break
                s = np.vstack((history[0], history[1], history[2], history[3]))    
                memory.append((s[None, :], action, reward)) 
                frame = self.process_state(obs) 
                frame = frame[:, :, None] 
                history = history[1:] + [frame]
             
            if len(history) == 0:
                # if the last state is terminal
                R = 0
            else:
                # else, compute next state value
                state = np.vstack((history[0], history[1], history[2], history[3]))
                R, action = self.select_action(state[None, :])

            size = len(memory)
            if size > 0:
                # setup parameters from memory
                self.actions, self.rewards = np.empty(self.num_actions), np.empty(1) 
                s = np.empty(memory[0][0].shape)

                for i in range(size-1, -1, -1):
                    R = memory[i][2] + self.gamma*R
                    act = np.zeros(self.num_actions, dtype=np.int) 
                    act[memory[i][1]] = 1
                    s = np.vstack((s, memory[i][0]))
                    self.actions = np.vstack((self.actions, act))
                    self.rewards = np.vstack((self.rewards, R)) 
            
                s = s[1:, :]
                self.actions = self.actions[1:, :]
                self.rewards = self.rewards[1:, :]

                # train the network
                self.network.fit(s, np.empty((size, self.num_actions+1)), batch_size=size, verbose=0)
    
    def evaluate(self, env, num_episodes, max_episode_length=1000000):
        """Test the agent with a provided environment.
        
        - Input
          - env: gym.Env
            This is the Atari environment. Need to wrap the
            environment using the wrap_atari_env function in the
            utils.py
          - num_episodes: int
            How many samples to perform.
          - max_episode_length: int
            How long a single episode should last before the agent

        - Output
          - total: float
            the cumulative rewards from all episodes
        """
        total = 0.0 

        for episode in range(num_episodes):
            obs = env.reset()
            s = self.process_state(obs)
            s = s[:, :, None]
            history = [s]
            action = env.action_space.sample()

            for _ in range(3):
                obs, reward, done, info = env.step(action)
                s = self.process_state(obs)
                s = s[:, :, None]
                history.append(s) 
            for t in range(max_episode_length):
                #env.render()
                state = np.vstack((history[0], history[1], history[2], history[3]))
                _, action = self.select_action(state[None,:])
                observation, reward, done, info = env.step(action)
                state = self.process_state(observation)
                state = state[:, :, None]
                history = history[1:]
                history.append(state)
                total += reward
                if done:
                    break

        return total
        
