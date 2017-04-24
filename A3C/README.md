We implement the A3C algorithm based on the following paper for the Atari game Enduro.

Mnih, Volodymyr, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and 
Koray Kavukcuoglu. "Asynchronous methods for deep reinforcement learning." 
In International Conference on Machine Learning, pp. 1928-1937. 2016.

Our implementation is based on tensorflow and keras, and we choose to share the network in multiple threads instead of creating one 
local network for each thread. 
