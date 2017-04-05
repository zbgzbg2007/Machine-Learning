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
from keras import backend as K
from agent import Agent
from scipy import ndimage

def create_model(input_shape, num_actions, model_type, window=4):  
    """Create the network model by Keras.  

    - Input
      - window: int
        Each input to the network is a sequence of frames. This value
        defines how many frames are in the sequence.
      - input_shape: tuple(int, int)
        The expected input image size.
      - num_actions: int
        Number of possible actions. Defined by the gym environment.
      - model_type: str 
        Possible choices could be: 'linear', 'deep', 'dueling', and we 
        construct linear model, deep Q-network or dueling Q-network respectively.

    - Output
      - keras.models.Model
        The Q-model.
    """
    # input dimension
    in_dim = (input_shape[0] * window, input_shape[1], 1)

    if model_type != 'dueling':
        model = Sequential()

        if model_type == 'linear':
            # Construct a linear Q-network
            model.add(Flatten(input_shape=in_dim))
            model.add(Dense(units=num_actions))

        elif model_type == 'deep':
            # Construct a deep Q-network  
            model.add(Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), activation='relu', input_shape=in_dim))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), activation='relu'))
            model.add(BatchNormalization())
            model.add(Flatten())
            model.add(Dense(units=256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(units=num_actions))

    else:
        # Construct a dueling Q-network
        inputs = Input(shape=in_dim)
        x = Conv2D(filters=32, kernel_size=(8,8), strides=(4,4), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(4,4), strides=(2,2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
        x = BatchNormalization()(x)
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


    # setup parameters
    # create the model and the agent
    # run fit method.
   
    env = gym.make(args.env)
    num_actions = env.action_space.n
    replay = True # apply replay memory or not
    double = False # apply double network or not

    model_type = 'deep' 
    model1 = create_model((105, 80), num_actions, model_type=model_type)  
    model2 = create_model((105, 80), num_actions, model_type=model_type)  
    preprocessor = core.Preprocessor()
    if replay:
        mem_size = 40000#1000000
        window = 4
        mem = core.ReplayMemory(mem_size, window)
    else: 
        mem = list()
    p = policy.LinearDecayGreedyEpsilonPolicy(1, 0.01, 500, num_actions)
    agent = Agent(
                 model_type=model_type,
                 q_network=model1,
                 preprocessor=preprocessor,
                 memory=mem,
                 policy=p,
                 gamma=0.99,
                 target_update_freq=1000,
                 num_burn_in=4000,
                 train_freq=10,
                 batch_size=32,
                 replay=replay,
                 target_network=model2,
                 double=double)
    agent.compile()
    agent.fit(env, num_iterations=400)
    agent.q_network.save_weights('last-weights')



if __name__ == '__main__':
    main()
