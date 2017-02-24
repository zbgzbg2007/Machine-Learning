import numpy as np

from layers import *
from layer_utils import *
from fast_layers import *

class ConvNet(object):
  '''
  A convolutional network with the following architecture:
  
  [conv-relu-conv-relu-pool] x N - [affine] x M - [softmax]
  
  The network operates on minibatch of shape (N, C, H, W), consisting
  of N images, each with C channels, height H and width W.
  
  Batch normalization is applied for each convolutional layer and each affine layer.
  L2 regularization is used.
  '''

  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 32], filter_size=3, 
               num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32,
               hidden_dim=[100, 100]):
    '''
    Initialize a network
   
    - Input:
      - input_dim: Tuple gives the shape of input
      - num_filters: List of numbers giving the number of filters 
                     in each convolutional layer, the length should be even
      - filter_size: Size of filters in convolutional layers
      - num_classes: Number of classes for training
      - weight_scale: Scalar giving standard deviation for weights initialization
      - reg: L2 regularization scalar (not used for convolutional layers)
      - dtype: numpy datatype to use in the network
      - hidden_dim: List of numbers giving the dimension of each affine layer
    

    '''

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn_params = []

    # Number of convolutional layers and affine layers
    self.num_conv = len(num_filters) 
    self.num_affine = len(hidden_dim) + 1

    # Initialize weights and bias for convolutional layers
    # Weights are sampled from standard normal distribution and scaled by weight_scale
    # Weights for each convolutional layer have shape (F, C, H, W)
    # Bias are zeros
    
    
    num_filters.insert(0, input_dim[0])
    for i in range(self.num_conv):
      self.params['W'+str(i)] = np.random.randn(num_filters[i+1], num_filters[i], filter_size, filter_size) * weight_scale
      self.params['b'+str(i)] = np.zeros(num_filters[i+1])

      # Parameters for batch normalziation: gamma is initialized to be one, beta zero
      self.params['gamma'+str(i)] = np.ones(num_filters[i+1])
      self.params['beta'+str(i)] = np.zeros(num_filters[i+1])
      self.bn_params.append({'mode': 'train'})
    
    # Initialize weights and bias for affine layers
    # Weights are sampled from standard normal distribution and scaled by weight_scale
    # Weights have shape (hidden_dim[i], hidden_dim[i+1])
    # Bias are zeros

    # The input for the first affine layer has shape (N, F', H1, W1)
    # F' is the number of filters in the last layer 
    # H1 and W1 can be obtained from input_dim[1] and input_dim[2] divided by 4**num_conv
    H1, W1 = input_dim[1] / 2**(len(num_filters)/2), input_dim[2] / 2**(len(num_filters)/2)
    hidden_dim.insert(0, num_filters[self.num_conv] * H1 * W1)
    hidden_dim.append(num_classes)
    i = self.num_conv - 1
    for j in range(self.num_affine):
      i += 1
      self.params['W'+str(i)] = np.random.randn(hidden_dim[j], hidden_dim[j+1]) * weight_scale
      self.params['b'+str(i)] = np.zeros(hidden_dim[j+1])
      
    
    # Parameters needed for convolutional layer
    self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
   
    # Parameters needed for pooling layer
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    '''
    Compute the loss and gradient for the network.
    Batch normalization is applied.
 
    - Input:
      - X: Input data of shape (N, C, H, W), where (C, H, W) is the same as input_dim
      - y: The labels of shape (N, ) for X. 

    - Output:
      - loss: If y is not None, loss is returned as the value of lose function defined by SoftMax
      - grads: If y is not None, grads is returned as the gradients for parameters in the network, including weights and bias
      - scores: If y is None, scores are returned in a list as the result for prediction
    '''
    scores = None
    cache = {}
    out = X

    if y is None:
      mode = 'test'
    else:
      mode = 'train'
 
    for bn in self.bn_params:
      bn['mode'] = mode

    # forward pass for convolutional layers
    for i in range(self.num_conv):
      if i % 2 == 0:
        out, cache[i] = conv_bn_relu_forward(out, self.params['W'+str(i)], self.params['b'+str(i)], self.conv_param, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i])
      else:
        out, cache[i] = conv_bn_relu_pool_forward(out, self.params['W'+str(i)], self.params['b'+str(i)], self.conv_param, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i], self.pool_param) 


    # forward pass for affine layers
    i = self.num_conv - 1
    for j in range(self.num_affine):
      i += 1
      out, cache[i] = affine_forward(out, self.params['W'+str(i)], self.params['b'+str(i)])

    scores = out

    if y is None:
      return scores

    loss, grads = 0, {}
    reg = self.reg
    
    loss, dout = softmax_loss(scores, y)
    
    # backward pass for affine layers
    for j in range(self.num_affine):
      dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dout, cache[i])
      loss += 0.5 * reg * np.sum(self.params['W'+str(i)] ** 2)
      grads['W'+str(i)] += reg * self.params['W'+str(i)]
      i -= 1

    # backward pass for convolutional layers
    for j in range(self.num_conv):
      if j % 2 == 0:
        dout, grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = conv_bn_relu_pool_backward(dout, cache[i])
      else:
        dout, grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = conv_bn_relu_backward(dout, cache[i])
      i -= 1

    return loss, grads
       


def conv_bn_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
  '''
  Customer layer for a convolution, a batch normalization and a ReLU
  
  - Input:
    - x: Input data of shape (N, D, H, W)
    - w: Weights for convolution
    - b: Bias for convolution
    - conv_param: Parameters for convolution
    - gamma: Scale scalar for batch normalization
    - beta: Shift scalar for batch normalization
    - bn_param: Parameters for batch normalization

  - Output:
    - out: Output of the layer, with shape (N, D, H1, W1)
    - cache: Intermediate information needed for backward propagation
  '''
  out, cache1 = conv_forward_fast(x, w, b, conv_param)
  out, cache2 = spatial_batchnorm_forward(out, gamma, beta, bn_param)
  out, cache3 = relu_forward(out)
  cache = (cache1, cache2, cache3)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  '''
  Backward pass for customer layer
  
  - Input:
    - dout: Upstream derivatives
    - cache: Intermediate information needed for backward propagation

  - Output:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient w.r.t. b
    - dgamma: Gradient w.r.t. gamma
    - dbeta: Gradient w.r.t. beta

  '''
  cache1, cache2, cache3 = cache
  dout = relu_backward(dout, cache3)
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout, cache2)
  dx, dw, db = conv_backward_fast(dout, cache1)

  return dx, dw, db, dgamma, dbeta


def conv_bn_relu_pool_forward(x, w, b, conv_param, gamma, beta, bn_param, pool_param):

  '''
  Customer layer for a convolution, a batch normalization, a ReLu and a pooling 

  - Input:
    - x: Input data of shape (N, D, H, W)
    - w: Weights for convolution
    - b: Bias for convolution
    - conv_param: Parameters for convolution
    - gamma: Scale scalar for batch normalization
    - beta: Shift scalar for batch normalization
    - bn_param: Parameters for batch normalization
    - pool_param: Parameters for pooling

  - Output:
    - out: Output of the layer, with shape (N, D, H1, W1)
    - cache: Intermediate information needed for backward propagation
  '''
  out, cache1 = conv_forward_fast(x, w, b, conv_param)
  out, cache2 = spatial_batchnorm_forward(out, gamma, beta, bn_param)
  out, cache3 = relu_forward(out)
  out, cache4 = max_pool_forward_fast(out, pool_param)
  cache = (cache1, cache2, cache3, cache4)
  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  '''
  Backward pass for customer layer
  
  - Input:
    - dout: Upstream derivatives
    - cache: Intermediate information needed for backward propagation

  - Output:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient w.r.t. b
    - dgamma: Gradient w.r.t. gamma
    - dbeta: Gradient w.r.t. beta

  '''
  cache1, cache2, cache3, cache4 = cache
  dout = max_pool_backward_fast(dout, cache4)
  dout = relu_backward(dout, cache3)
  dout, dgamma, dbeta = spatial_batchnorm_backward(dout, cache2)
  dx, dw, db = conv_backward_fast(dout, cache1)
  return dx, dw, db, dgamma, dbeta
