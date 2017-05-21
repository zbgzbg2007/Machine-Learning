"""Main agent class
   terminal state is not stored in memory
"""
import tensorflow as tf
import numpy as np
from keras.optimizers import RMSprop
import core
import policy
import sys
from scipy import ndimage
import cv2
class Agent:
    """Class implementing the agent.

    - Parameters
      - model_type: str
        The architecture of your model.
      - q_network: keras.models.Model
        The Q-network model.
      - memory: core.ReplayMemory 
        The replay memory 
      - policy: Policy
        The policy for agent
      - gamma: float
        Discount factor.
      - window: int
        The number of frames as input
      - update_freq: int 
        Frequency to update q-network
      - target_update_freq: int
        Frequency to update the target network. 
      - num_burn_in: int
        Before beginning updating the Q-network, the replay memory has
        to be filled up with some number of samples. This is the number.
      - batch_size: int
        How many samples in each minibatch.
      - output: str
        The name of output file
    """
    def __init__(self,
                 model_type,
                 q_network,
                 memory,
                 policy,
                 gamma,
                 window,
                 update_freq,
                 target_update_freq,
                 num_burn_in,
                 batch_size,
                 ** kwargs):
        self.model_type = model_type 
        self.q_network = q_network
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.window = window
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq * update_freq
        self.num_burn_in = num_burn_in
        self.batch_size = batch_size
        self.target_network = kwargs.pop('target_network', None)
        self.double = kwargs.pop('double', False)
        self.output = kwargs.pop('output', self.model_type+'-output')

    def mean_huber_loss(self, y_true, y_pred, max_grad=1.):
        """Calculate the mean huber loss, a robust loss function.
        See https://en.wikipedia.org/wiki/Huber_loss

        - Input
          - y_true: np.array, tf.Tensor
            Target value.
          - y_pred: np.array, tf.Tensor
            Predicted value.
          - max_grad: float, optional
            Positive floating point value. Represents the maximum possible
            gradient magnitude.

        - Output
          - tf.Tensor
            The huber loss.
        """
        err = tf.abs(y_true - y_pred, name='abs')
        delta = tf.constant(max_grad, name='max_grad')
        less = 0.5 * err**2
        more = delta * (err - 0.5 * delta)
        return tf.reduce_mean(tf.where(err<=delta, less, more))
    

    def compile(self):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.
        """
        opt = RMSprop(lr=0.00025, rho=0.95)
        #self.q_network.load_weights('deep-double-output7-weights')
        self.q_network.compile(optimizer=opt, loss= 'mean_squared_error')
        #self.q_network.compile(optimizer=opt, loss= self.mean_huber_loss)
        weights = self.q_network.get_weights()

        #self.target_network.load_weights('deep-double-output7-weights')
        self.target_network.set_weights(weights)
        self.target_network.compile(optimizer=opt, loss=self.mean_huber_loss)
        
    

    def select_action(self, batch, policy, **kwargs):
        """Select the action based on the current state in the batch.

       
        - Input
          - batch: list of states 
          - policy: policy.policy
            The policy according to which to choose action
          - is_training: bool, optional
            If true then parameter in the policy will be decayed. Only works for 
            epsilon decay policy
        
        - Output
          - action: list of actions
        """
        is_training = kwargs.pop('is_training', True)
           
        p = policy
        q_val = self.q_network.predict(batch, batch_size=len(batch))
        action = list()
        for q in q_val:
            action.append(p.select_action(q_values=q, is_training=is_training))
        return action

    def update_network(self):
        """Update the Q-network.  """
        # obtain batch from replay memory 

        imgs, actions, rewards, terminal = self.memory.sample(self.batch_size)
        size = actions.shape[0]

        # obtain data from batch
        inputs = imgs[:, :-1, ...]
        nextinputs = imgs[:, 1:, ...]

        # calculate target values for the target network
        next_y = self.target_network.predict(nextinputs, batch_size=self.batch_size)
        
        # apply double DQN or not
        if self.double:
            q_val = self.q_network.predict(nextinputs, batch_size=self.batch_size)
            target_actions = np.argmax(q_val, axis=1)
            outputs = next_y[range(next_y.shape[0]), target_actions]
        else:
            outputs = np.max(next_y, axis=1)

        discounted_outputs = outputs * self.gamma
        discounted_outputs[terminal] = 0.
        targets = rewards + discounted_outputs

        target_y = self.q_network.predict(inputs, batch_size=self.batch_size)
        target_y[range(target_y.shape[0]), actions] = targets 
    
        # do gradient descent
        self.q_network.fit(inputs, target_y, batch_size=self.batch_size, verbose=0)


    def process_state_for_network(self, obs):
        """Preprocess the given state before giving it to the network.
           Convert the image to greyscale and downsample the image.  

        - Input
          - obs: np.ndarray
            A single observation from an environment.

        - Output
        -------
          - processed_state: np.ndarray
            The state after processing. Can be modified in anyway.

        """
        #state = obs[:160, ...] / 255.
        # convert RGB image into grayscale
        state = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        # resize image into 80 * 80 
        state = cv2.resize(state[:160, :], (80, 80)) / 255.0
        return state



    def fit(self, env, num_iterations, max_episode_length=5000):
        """Fit the model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        - Input
          - env: gym.Env
            This is the Atari environment. 
          - num_iterations: int
            How many samples/updates to perform.
          - max_episode_length: int
            How long a single episode should last before the agent
            resets. Can help exploration.
        """

        step = 0 # record how many updates are taken
        history = np.empty((self.window, 80, 80), dtype='float32') # history stores the last self.window number of  processed frames as input for network
        for episode in range(num_iterations):
            # periodically evaluate the performance
            if (episode+1) % 40 == 0: 
                total = self.evaluate(env, 4, max_episode_length) 
                with open(self.output, 'a') as out:
                    print >>out, total
                print 'Evaluate: ', total
            if (episode+1) % 40 == 0:
                self.q_network.save_weights(self.output+'-weights', 'w')

            print 'Episode ', episode+1
            obs = env.reset() 
            for i in range(self.window):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action) 
                state = self.process_state_for_network(obs)
                history[i] = state
            index = 0 # index for history 
            for t in range(max_episode_length):
                state = history
                action = self.select_action(state[None, ...], self.policy)[0]
                t += 1
                obs, reward, done, info = env.step(action) 
                        
                state = self.process_state_for_network(obs) 
                history[index%self.window] = state
                index += 1

                # update replay memory
                reward = np.clip(reward, -1., 1.)
                self.memory.append(history[(index-2)%self.window], action, reward, done)
                    
                if len(self.memory) >= self.num_burn_in: 
                    # if there are enough frames in replay memory, 
                    step += 1
                    if step % self.update_freq == 0:
                        # update Q-network
                        self.update_network()
                    if step % self.target_update_freq == 0:
                        # if enough steps taken, update target network
                        weights = self.q_network.get_weights()
                        self.target_network.set_weights(weights)

                if done:
                    break


    def evaluate(self, env, num_episodes, max_episode_length=5000):
        """Test the agent with a provided environment.
        
        If you have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        - Input
          - env: gym.Env
            This is the Atari environment. Need to wrap the
            environment using the wrap_atari_env function in the
            utils.py
          - num_iterations: int
            How many samples/updates to perform.
          - max_episode_length: int
            How long a single episode should last before the agent

        - Output
          - total: float
            the cumulative rewards from all episodes
        """
        total = 0.0 
        p = policy.GreedyEpsilonPolicy(0.05, env.action_space.n) 

        history = np.empty((self.window, 80, 80), dtype='float32')
        for episode in range(num_episodes): 
            obs = env.reset()
            index = 0
            for i in range(self.window):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action) 
                state = self.process_state_for_network(obs)
                history[i] = state
                
            for t in range(max_episode_length):
                #env.render()
                state = history
                action = self.select_action(state[None, ...], p, is_training=False)
                obs, reward, done, info = env.step(action)
                total += reward
                if done: 
                    break
                state = self.process_state_for_network(obs)
                history[index%self.window] = state
                index += 1
        
        return total
