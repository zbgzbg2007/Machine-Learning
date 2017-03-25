# Assignment 2

This assignment includes 4 problems:

- Q1: Image Captioning with Vanilla RNNs (RNN_Captioning.ipynb)

  We implemented layers for vanilla RNN, including the forward and backward passes for a single timestep, for the processing the 
  entire sequence, and word embedding. We combine those layers to build an image captioning model with the following architecture:
  ```
  features ->(affine) initial hidden state  -> RNN processing entire sequence ->(affine) scores for vocabulary at each timestep ->(softmax) -> loss 
  captions ->(word embedding) input for RNN ->
  ```
  For test-time, we implemented a sample function to sample the words for caption.
  
- Q2: Image Captioning with LSTMs (LSTM_Captioning.ipynb)

  We implemented the forward and backward passes for the Long-Short Term Memory RNN, and an LSTM by those layers. The architecture
  is similar to the vanilla RNN. We also implemented the sample funciton for LSTM.

- Q3: Image Gradients: Saliency maps and Fooling Images (ImageGradients.ipynb)

  We Use a pretrained CNN model to compute the gradient of the image and then compute class saliency maps as described in
  Section 3.1 of [1].
  
  We also use image gradients to generate "fooling images" as discussed in [2]. Given an image and a target class, 
  we perform gradient ascent over the image to maximize the target class, stopping when the network classifies 
  the image as the target class. Implement the following function to generate fooling images.
  
  [1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising Image
  Classification Models and Saliency Maps", ICLR Workshop 2014.
  
  [2] Szegedy et al, "Intriguing properties of neural networks", ICLR 2014.
  
- Q4: Image Generation: Classes, Inversion, DeepDream (ImageGeneration.ipynb)

  Starting from a random noise image, we generate an image that the network will recognize as the target class by 
  performing gradient ascent on the target class. We also add L2 regularization to our generation. 
  
  We also reconstruct an image from its feature representation. That is, given an image and a layer in the model, we construct 
  another image that has similar feature representation at that layer. This is done by by solving an optimization problem 
  using image gradients from the pretrained network. 

  Finally, we implemented a simple version of DeepDream. For a chosen layer, the idea is to set the gradients the same values
  as the activations. This is the same as to maximize the L2 norma of the activations in that layer and amplifies the
  activations in that layer. 
  We also tried another idea discribed at the end of the page https://github.com/google/deepdream/blob/master/dream.ipynb .
  That is to guide the partterns in the result by another guide image. The method is to maximize the dot product between the 
  activations of our current image and their best matching activations from the guide image. This is the result from our 
  implementation.
  
  ![](https://github.com/zbgzbg2007/Machine-Learning/blob/master/CS231n/assignment3/dream-sky.png)
