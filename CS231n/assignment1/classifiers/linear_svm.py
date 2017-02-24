import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = dW.T
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
	dW[j] += X[i]
	dW[y[i]] -= X[i]
  
  dW = dW.T
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  score = X.dot(W)
  num_train = X.shape[0]
  score = (score.T - score[np.arange(num_train), y]).T
  score += 1
  score[np.arange(num_train), y] = 0
  score = np.maximum(score, 0)  #almost forgot
  loss = np.sum(score) / num_train
  loss += 0.5 * reg * np.sum(W * W) #forgot this
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  score = np.minimum(np.ceil(score), 1) # missed ceil 
  dW = X.T.dot(score)
    
  Z = np.sum(score, axis = 1)
  Z = np.reshape(Z, (Z.shape[0], -1)) * (-X) 
  
  num_classes = W.shape[1]
  Y = np.zeros((num_train, num_classes))
  Y[range(num_train), y] = 1

  dW += Z.T.dot(Y)
  
  """
	This is crazy. Need to explain this.
	Gradient comes from two parts. The first is for each column j 
	where j != y_i, the second is for each column j where j == y_i.
	For the first part, we have 

	dW_{ij} = \sum_k 1(score_{kj} > 0) * X_{ki}
		= \sum_k X_{ik} * score_{kj}       
		= X.T.dot(score)

	This is because score_{ky_i} == 0 and each nonzero in score is 1.
	For the second part, we have 

	dW_{ij} = \sum_{k: y[k] == j} \sum_{l != j} 
		  1(score_{kl} > 0) * (-X_{ki})
		= \sum_{k: y[k] == j} (\sum_l 1(score_{kl} > 0)) * (-X_{ki})
		= \sum_{k: y[k] == j} Sum_k * (-X_{ki})
	
	Let Z be the sum of each row of score: Z_k = \sum_l 1(score_{kl} > 0).
	Then Z = Z.T * (-X) has kth row as Sum_k * (-X_k).
	Let Y be the array where Y_{ij} = 1 if y[i] = j.
	Then Z.T.dot(Y) contains jth column as sum of rows of Z whose 
	class labels y are all equal to j.
	This is the second part for dW.
	

  """

  dW /= num_train 
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
