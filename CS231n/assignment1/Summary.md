# Assignment 1

This assignment includes 5 problems:

- Q1: k-Nearest Neighbor classifier
  
  We implemented the functions for compute distances for kNN and the function for prediction in 
  classifiers/k_nearest_neighbor.py. We have different to compute the distances: trivially compare two points (two loops), partially
  vectorized method (one loop) and fully vectorized method (no loop). **_This shows the efficiency of the vectorized method._**
  
  We also implemented cross-validation to choose the best value for hyperparameter k in knn.ipynb.

- Q2: Training a Support Vector Machine 

  We implemented the loss function for SVM that computes the loss and gradient in two ways in classifiers/linear_svm.py.
  The first way is to compute the values by one loop and the other is to compute those in fully vectorized way. The second method
  is about 10X faster but needs some calculus. We also implemented the predict function for SVM and SGD in linear_classifier.py to 
  train the classifier. In svn.ipynb, we use validation set to tune the learning rate and regularization strength.
  
  On the preprocessing: any preprocessing statistics (like data mean) must be computed only form the training data and then applied
  to other split (train, validation and test).

- Q3: Implement a Softmax classifier 

  We implemented a native softmax loss function by nested loop and a vectorized softmax loss function in classifiers/softmax.py.
  We checked our implementation with numerical gradient and train it by SGD. In softmax.ipynb, we use validation set to tune the hyperparamenters. 
  
  Since the initial weights are random, the socre of each class should be similar to others. So the initial loss should be close to 
  -log(1/C) for C classes, or log(C).

- Q4: Two-Layer Neural Network 

  We implemented a loss function classifiers/neural_net.py for a two-layer neural network to compute the loss and 
  gradient. The loss is computed in the forward pass and gradient is computed in the backward pass. We also tuen the hyperparamenters
  by using validation set in two_layer_net.ipynb.
  
  We tried PCA preprocessing, but it seems not very helpful.

- Q5: Higher Level Representations: Image Features 

  This part is all about the training techniques and hyperparameter search. Although the lecture recommand random search
  for hyperparameters, I tried from coarse to fine search. That is, first pick some values with big difference and then shrink
  the difference for those good values. I tried [1, 10, 0.1, 0.01, 100] and then [1.1, 1, 0.9] for learning rate. 
  When the training makes no progress and the loss does not change by much, the reason I find is that the learning rate is 
  too small; when the loss jumps too big if the batch size is small; when the loss increases, the reason is the learning rate
  is too big. For the two-layer neural network, the regularization should be small.
  

