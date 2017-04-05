"""Main agent class"""
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
import core
import policy
import sys
class Agent:
    """Class implementing the agent.

    - Parameters
      - model_type: str
        The architecture of your model.
      - q_network: keras.models.Model
        The Q-network model.
      - preprocessor: core.Preprocessor
        The preprocessor class. See the associated classes for more
        details.
      - memory: core.ReplayMemory or a list of transitions
        The replay memory if you apply replay memory (replay is True); 
        otherwise (replay is False) a list storing some transitions to update once.
      - gamma: float
        Discount factor.
      - target_update_freq: int
        Frequency to update the target network. 
      - num_burn_in: int
        Before beginning updating the Q-network, the replay memory has
        to be filled up with some number of samples. This is the number.
      - train_freq: int
        Frequency to actually update your Q-Network. Only works if replay is False.
      - batch_size: int
        How many samples in each minibatch.
      - replay: bool
        If apply experience replay.
    """
    def __init__(self,
                 model_type,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 ** kwargs):
        self.model_type = model_type 
        self.q_network = q_network
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.replay = kwargs.pop('replay', False)
        self.target_network = kwargs.pop('target_network', None)
        self.double = kwargs.pop('double', False)

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
        opt = Adam(lr=0.0001)
        self.q_network.compile(optimizer=opt, loss=self.mean_huber_loss)
        if self.replay:
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
            action.append(p.select_action(q))
        return action

    def update_network(self):
        """Update the Q-network.

        """
        p = policy.GreedyPolicy()
        if self.replay:
            # obtain batch from replay memory 
            batch = self.memory.sample(self.batch_size)
            batch = self.preprocessor.process_batch(batch)
            size = len(batch)
        else:
            # obtain batch from memory and process batch
            n = len(self.memory)
            batch = list()
            for i in range(3, n-1):
                state, a, r, nexts, done = self.memory[i]       
                state, nexts = self.memory[i-3][0], self.memory[i-3][3]
                for x in range(2, -1, -1):
                    state = np.vstack((state, self.memory[i-x][0]))
                    if done == False: 
                        nexts = np.vstack((nexts, self.memory[i-x][3]))
                    else:
                        nexts = None
                batch.append((state, a, r, nexts, done))
            size = len(batch)-1

        # obtain data from batch
        inputs, actions, outputs, nextinputs, terminal = batch[0]
        inputs = inputs[None, :]
        if terminal == False:
            nextinputs = nextinputs[None, :]
            terminal = self.gamma
        else:
            nextinputs = inputs
            terminal = 0
        for i, sample in enumerate(batch):
            if i == 0:
                continue
            state, action, reward, nexts, is_terminal = sample
            state = state[None, :] 
            inputs = np.vstack((inputs, state))
            actions = np.hstack((actions, action))
            outputs = np.hstack((outputs, reward))
            if is_terminal == False:
                nexts = nexts[None, :]   
                nextinputs = np.vstack((nextinputs, nexts))
                terminal = np.vstack((terminal, self.gamma))
            else:
                nextinputs = np.vstack((nextinputs, state))
                terminal = np.vstack((terminal, 0))

        # calculate target values for the target network
        next_y = self.target_network.predict(nextinputs, batch_size=size)
        next_y = next_y * terminal * self.gamma
        target_y = self.q_network.predict(inputs, batch_size=size)

        # apply double DQN or not
        if self.double:
            target_actions = np.argmax(target_y, axis=1)
            outputs = outputs + next_y[range(next_y.shape[0]), target_actions]
        else:
            outputs = outputs + np.max(next_y, axis=1)

        target_y[range(target_y.shape[0]), actions] = outputs
            
        # do gradient descent
        self.q_network.fit(inputs, target_y, batch_size=size, verbose=0)





    def fit(self, env, num_iterations, max_episode_length=1000000):
        """Fit the model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        - Input
          - env: gym.Env
            This is the Atari environment. Need to wrap the
            environment using the wrap_atari_env function in the
            utils.py
          - num_iterations: int
            How many samples/updates to perform.
          - max_episode_length: int
            How long a single episode should last before the agent
            resets. Can help exploration.
        """

        step = 0 # record how many steps are taken

        for episode in range(num_iterations):
            # periodically evaluate the performance
            if episode % 10 == 0:
                total = self.evaluate(env, 8)
                with open(self.model_type+'-output', 'w') as out:
                    print >>out, total

            observation = env.reset()
            state = self.preprocessor.process_state_for_network(observation)
            state = state[:, :, None]

            # history stores the last 4 frames
            history = [np.zeros(state.shape), np.zeros(state.shape), np.zeros(state.shape), state]

            for t in range(max_episode_length):
                #env.render()
                # show progress
                sys.stdout.write('\r')
                sys.stdout.write("[%-60s] %d%%" % ('='*(60*(t+1)/max_episode_length), (100*(t+1)/max_episode_length)))
                sys.stdout.flush()
                sys.stdout.write(", step %d "% (t+1))
                sys.stdout.flush()

                state = np.vstack((history[0], history[1], history[2], history[3]))
                action = self.select_action(state[None,:], self.policy)
                next_observation, reward, done, info = env.step(action)

                if self.replay:
                    # if replay memory is used, update replay memory
                    state = self.preprocessor.process_state_for_memory(observation)
                    self.memory.append(state, action, reward, done)
                    state = self.preprocessor.process_state_for_network(next_observation)
                    state = state[:, :, None]
                else:
                    # if replay memory is not used, update memory
                    nexts = self.preprocessor.process_state_for_network(next_observation)
                    nexts = nexts[:, :, None]
                    self.memory.append((history[3], action, reward, nexts, done))
                    state = nexts

                history = history[1:]
                history.append(state)

                step += 1  

                if self.replay: 
                    if step > self.num_burn_in: 
                        # if there are enough frames in replay memory, 
                        # update Q-network
                        #self.update_network()
                        if step % self.target_update_freq == 0:
                            # if enough steps taken, update target network
                            weights = self.q_network.get_weights()
                            self.target_network.set_weights(weights)
                else:
                    # replay memory is not used
                    if step % self.train_freq == 0:
                        # after collecting some data, 
                        # update Q-network and reset memory 
                        self.update_network()
                        self.memory = self.memory[-3:]
                if done:
                    break


    def evaluate(self, env, num_episodes, max_episode_length=1000000):
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
        greedy = policy.GreedyPolicy()

        for episode in range(num_episodes):
            #print total
            observation = env.reset()
            state = self.preprocessor.process_state_for_network(observation)
            state = state[:, :, None]
            history = [np.zeros(state.shape), np.zeros(state.shape), np.zeros(state.shape), state]
            for t in range(max_episode_length):
                env.render()
                state = np.vstack((history[0], history[1], history[2], history[3]))
                action = self.select_action(state[None,:], greedy)
                observation, reward, done, info = env.step(action)
                state = self.preprocessor.process_state_for_network(observation)
                state = state[:, :, None]
                history = history[1:]
                history.append(state)
                total += reward
                if done:
                    break

        return total
