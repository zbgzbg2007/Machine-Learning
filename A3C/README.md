We implement the A3C algorithm based on the following paper for the Atari game Enduro.

Mnih, Volodymyr, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and 
Koray Kavukcuoglu. "Asynchronous methods for deep reinforcement learning." 
In International Conference on Machine Learning, pp. 1928-1937. 2016.

We have two versions of implementations. 
The first is based on tensorflow and keras.
We apply multiprocessing to run multiple games at the same time and collect input data from those games.
Our main process contains an agent which will decide the actions for those games and be trained by the data from those games.
The second is based on pytorch and we apply shared memory for the model so all the processes can train the model and this is
closer to the implementation in the paper.

