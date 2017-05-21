"""Core classes."""

import numpy as np
from scipy import ndimage
import random
import heapq
    


class ReplayMemory:
    """Replay memory class.

       Implemented using a list as a ring buffer.
       Each sample consists of (state, action, reward).

    - Methods
      - append(state, action, reward, is_terminal)
        Add a sample to the replay memory. 
      - sample(batch_size)
        Return list of samples from the memory. 
      - clear()
        Reset the memory. 
    """
 
    def __init__(self, max_size, window_length):
        """Setup memory.
        - Input
          - max_size: int
            The maximum size of memory
          - window_length: int
            The number of frames as input

        """

        self.max_size = max_size
        self.imgs = np.empty((max_size, 80, 80), dtype='float32') 
        self.actions = np.empty(max_size, dtype='uint8') 
        self.rewards = np.empty(max_size)
        self.terminal = np.empty(max_size, dtype='bool') 
        self.index = 0 # this index always be available
        self.size = 0
        self.window = window_length

    def __len__(self):
        return min(self.size, self.max_size)

    def append(self, state, action, reward, is_terminal):
        '''Append a sample into memory
 
        '''
        if self.size < self.max_size:
            self.size += 1
        self.imgs[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminal[self.index] = is_terminal
        self.index = (self.index + 1) % self.max_size
 
        
    def sample(self, batch_size):
        '''Uniformly random sample some transitions from memory
        
        - Input
          - batch_size: int
            The number of samples
 
        - Output
          - batch: list of transitions (a tuple of 4 elements)
            A batch of sampled transitions 
        '''
        
        imgs = np.empty((batch_size, self.window+1, 80, 80), dtype='float32')
        act = np.empty(batch_size, dtype='int32')
        rwd = np.empty(batch_size, dtype='float32')
        tml = np.empty(batch_size, dtype='bool')
        start = 0 if self.size < self.max_size else self.index
        for i in range(batch_size):
            # sample an legal index: the first three frames are not terminal
            while True: 
                j = random.randint(start, start+self.size-self.window)
                indices = np.arange(j, j+self.window+1)
                if np.any(self.terminal.take(indices[:-2], mode='wrap')):
                    continue
                else:
                   index = indices[self.window-1]
                   imgs[i] = self.imgs.take(indices, axis=0, mode='warp')
                   act[i] = self.actions.take(index, mode='warp')
                   rwd[i] = self.rewards.take(index, mode='warp')
                   tml[i] = self.terminal.take(index, mode='warp')
                   break
                   
        return imgs, act, rwd, tml

    def clear(self):
        '''Reset the memory
        '''
        self.index = 0
        self.size = 0
        
