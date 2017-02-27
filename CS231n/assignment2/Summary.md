# Assignment 1

This assignment includes 4 problems:

- Q1: Fully-connected Neural Network 

  In layers.py, we implemented the forward pass and backward passes for affine layer, ReLU layer. In classifiers/fc_net.py we
  implemented a two-layer neural network and a fully-connected network with arbitrary number of hiden layers in the modular way 
  by using our layers. The network class contains two methods: initialization and loss function.
  We trained a three-layer network and a five-layer network by a small set. The deeper network seems more 
  sensitive. 
  
  In cs231n/optim.py, we implemented some update rules including the SGD+momentum, RMSProp and Adam update rules. The comparisons 
  shows that they are faster than the vanilla SGD. 
  

- Q2: Batch Normalization

  In layers.py, we implemented the forward and backward passes for batch normalization layer. And we added this layer into our 
  fully-connected network. For training set, BN is faster and more accurate and BN is not very sensetive to the weight 
  initialization by comparisons. We give more details about BN in the BN file in this folder.

- Q3: Dropout 

  We implemented the forward and backward passes for dropout layer in layers.py, and added it into our network. By doing some 
  experiments, it can be shown that dropout layer reduces the training accuracy but the accuarcy of validation set increases.

- Q4: ConvNet on CIFAR-10 

  We implemented the two passes for convolutional layer, max pooling layer, batch normalization for convolution layer in layers.py. 
  With these layers, we implemented a three-layer convolutional network in classifiers/cnn.py with the architecture:
  
  [conv - relu - 2x2 max pool] - [affine - relu] - [affine] - softmax.
  
  In xcnn.py, we also implmented another convolutional network with arbitrary number of layers with the architecture:
  
  [conv-relu-conv-relu-pool] x N - [affine] x M - softmax.

  Batch normalziation is applied between each convolutional layer and ReLU layer.
  
