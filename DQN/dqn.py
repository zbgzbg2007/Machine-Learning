"""Run Atari Environment with Agent."""
import argparse
import os
import random
import gym
import keras
import agent
import core
import policy
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Flatten, Input, Lambda,
                          Permute, BatchNormalization)
from keras.models import (Model, Sequential)
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from agent import Agent
from scipy import ndimage

def create_model(input_shape, num_actions, model_type, window=4):  
    """Create the network model by Keras.  

    - Input
      - input_shape: tuple(int, int)
        The expected input image size.
      - num_actions: int
        Number of possible actions. Defined by the gym environment.
      - model_type: str 
        Possible choices could be: 'linear', 'deep', 'dueling', and we 
        construct linear model, deep Q-network or dueling Q-network respectively.
      - window: int
        The number of frames as input

    - Output
      - keras.models.Model
        The Q-model.
    """
    # input dimension
    in_dim = (window, input_shape[0], input_shape[1])

    if model_type != 'dueling':
        model = Sequential()

        if model_type == 'linear':
            # Construct a linear Q-network
            model.add(Permute((2, 3, 1), input_shape=in_dim))
            model.add(Flatten())
            model.add(Dense(units=num_actions))

        elif model_type == 'deep':
            # Construct a deep Q-network  
            model.add(Permute((2, 3, 1), input_shape=in_dim))
            model.add(Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu'))
            #model.add(BatchNormalization())
            model.add(Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu'))
            model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
            #model.add(BatchNormalization())
            model.add(Flatten())
            model.add(Dense(units=512, activation='relu'))
            #model.add(BatchNormalization())
            model.add(Dense(units=num_actions))

    else:
        # Construct a dueling Q-network
        inputs = Input(shape=in_dim)
        x = Permute((2, 3, 1))(inputs)
        x = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
        #x = BatchNormalization()(x)
        x = Flatten()(x)
        val = Dense(units=512, activation='relu')(x)
        adv = Dense(units=512, activation='relu')(x)
        V = Dense(units=1, activation='relu')(val)
        A = Dense(units=num_actions, activation='relu')(adv)
        x = keras.layers.concatenate([V, A])
        # last layer applies the average of advantage function 
        # modification from the paper
        Q = Lambda(lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(num_actions, ))(x)
        model = Model(inputs=inputs, outputs=Q)

    return model




def main():  
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    #args.output = get_output_folder(args.output, args.env)

    # setup parameters
    # create the model and the agent
    # run fit method.
   
    env = gym.make(args.env)
    num_actions = env.action_space.n
    double = True # apply double network or not

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.25
    set_session(tf.Session(config=config))
    model_type = 'deep' 
    output = model_type+'-double-output'+'10'
    window = 4
    model1 = create_model((80, 80), num_actions, model_type=model_type, window=window)  
    model2 = create_model((80, 80), num_actions, model_type=model_type, window=window)  
    mem_size = 1000000
    mem = core.ReplayMemory(mem_size, window)
    p = policy.LinearDecayGreedyEpsilonPolicy(1., 0.1, 1000000, num_actions)
    #p = policy.GreedyEpsilonPolicy(.1, num_actions)
    agent = Agent(
                 model_type=model_type,
                 q_network=model1,
                 memory=mem,
                 policy=p,
                 gamma=0.99,
                 window=window,
                 update_freq=4,
                 target_update_freq=10000,
                 num_burn_in=50000,
                 batch_size=32, 
                 target_network=model2,
                 double=double,
                 output=output)
    agent.compile()
    agent.fit(env, num_iterations=4000, max_episode_length=250000)
    #rewards = agent.evaluate(env, num_episodes=10)
    #print rewards
    agent.q_network.save_weights(output+'-last-weights')



if __name__ == '__main__':
    main()
