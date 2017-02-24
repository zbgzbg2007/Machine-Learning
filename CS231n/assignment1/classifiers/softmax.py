import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  L_i = -log p_{y[i]}
  dL_i/dW_j = (p_j - [j == y[i]]) * X[i]

  """ 
  num_class = W.shape[1] 
  num_train = X.shape[0]
  dW = dW.T
  for z in range(num_train):
    f = X[z].dot(W)
    f -= np.max(f)
    f = np.exp(f)
    p = f / np.sum(f)
    loss -= np.log(p[y[z]])
    p[y[z]] -= 1
    for d in range(num_class):
        dW[d] += X[z] * p[d]
  dW = dW.T
  loss /= num_train
  dW /= num_train
  loss += np.sum(W*W) * 0.5 * reg
  dW += reg * W 

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 
  num_train = X.shape[0] 
  f = X.dot(W)
  f = (f.T-f[range(num_train), np.argmax(f, axis = 1)]).T
  f = np.exp(f)
  p = (f.T / np.sum(f, axis = 1)).T
  loss = np.sum(-np.log(p[range(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  p[range(num_train), y] -= 1
  dW = X.T.dot(p)
  dW /= num_train
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

