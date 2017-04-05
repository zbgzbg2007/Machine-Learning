"""Core classes."""

import numpy as np
from scipy import ndimage
import random

    
class Preprocessor:
    """Preprocessor class.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.  

    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.
           Convert the image to greyscale and downsample the image.  


        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        - Input
          - state: np.ndarray
            A single observation from an environment.

        - Output
        -------
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

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        - Input
          - state: np.ndarray
            A single state from an environmnet. 

        - Output
          - processed_state: np.ndarray
            The state after processing. 

        """
        return state.astype('uint8')

    def process_batch(self, samples):
        """Process batch of samples.

        Since the replay memory storage format is different than the
        network input, we need to apply this function to the
        sampled batch before running it through the update function.

        - Input
          - samples: a list of transitions (a tuple of 5 elements)
            - List of samples to process

        - Output
          - processed_samples: a list of transitions (a tuple of 5 elements)
            Samples after processing. 
        """

        batch = list()
        for s in samples:
            state, action, reward, nexts, is_terminal = s
            image = None

            # process consecutive frames for state
            for frame in state:
                prs_frame = self.process_state_for_network(frame)
                if image is None:
                    image = prs_frame[:, :, None]
                else:
                    image = np.vstack((image, prs_frame[:, :, None]))
            next_image = None
            if is_terminal == False:
                # if not terminal state, process frames for new state 
                for frame in nexts:
                    prs_frame = self.process_state_for_network(frame)
                    if next_image is None:
                        next_image = prs_frame[:, :, None]
                    else:
                        next_image = np.vstack((next_image, prs_frame[:, :, None]))
                 
            batch.append((image, action, reward, next_image, is_terminal))

        return batch





class ReplayMemory:
    """Replay memory class.

       Implemented using a list as a ring buffer.

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

        """

        self.max_size = max_size
        self.size = 0
        self.window = window_length
        self.memory = list()
        self.index = 0 # this index always be available

    def append(self, state, action, reward, is_terminal):
        '''Append a sample into memory
 
        '''
        if self.size < self.max_size:
            self.memory.append((state, action, reward, is_terminal))
            self.size += 1
        else:
            self.memory[self.index] = (state, action, reward, is_terminal)
        self.index = (self.index + 1) % self.max_size
 
        
    def sample(self, batch_size):
        '''Uniformly random sample some transitions from memory
        
        - Input
          - batch_size: int
            The number of samples
 
        - Output
          - batch: list of transitions (a tuple of 5 elements)
            A batch of sampled transitions 
        '''

        batch = list()
        for _ in range(batch_size):
            # sample an legal index
            i = random.randint(0, self.size-1)
            while not ( i != self.index%self.size and 
                       (i-1)%self.max_size != self.index%self.size and 
                       (i-2)%self.size != self.index%self.size and 
                       (i-3)%self.size != self.index%self.size and
                       ((i+1)%self.size != self.index%self.size or 
                       self.memory[i][3] == True)):
                i = random.randint(0, self.size-1)

            # collect consecutive frames for state
            state = list()
            state.append(self.memory[(i-3)%self.max_size][0]) 
            state.append(self.memory[(i-2)%self.max_size][0]) 
            state.append(self.memory[(i-1)%self.max_size][0]) 
            state.append(self.memory[(i)%self.max_size][0]) 
            
               
            s, action, reward, is_terminal = self.memory[i]
            # collect consecutive frames for new state
            if is_terminal == False:
                next_state = list()
                next_state.append(self.memory[(i-2)%self.max_size][0]) 
                next_state.append(self.memory[(i-1)%self.max_size][0]) 
                next_state.append(self.memory[(i)%self.max_size][0]) 
                next_state.append(self.memory[(i+1)%self.max_size][0]) 
            else:
                next_state = None

            batch.append((state, action, reward, next_state, is_terminal))

        return batch

    def clear(self):
        '''Reset the memory
        '''
        self.index = 0
        self.size = 0
        
