# Assignment 2

This assignment includes 4 problems:

- Q1: Image Captioning with Vanilla RNNs

  We implemented layers for vanilla RNN, including the forward and backward passes for a single timestep, for the processing the 
  entire sequence, and word embedding. We combine those layers to build an image captioning model with the following architecture:
  ```
  features ->(affine) initial hidden state  -> RNN processing entire sequence ->(affine) scores for vocabulary at each timestep ->(softmax) -> loss 
  captions ->(word embedding) input for RNN ->
  ```
  For test-time, we implemented a sample function to sample the words for caption.
  
- Q2: Image Captioning with LSTMs

  We implemented the forward and backward passes for the Long-Short Term Memory RNN, and an LSTM by those layers. The architecture
  is similar to the vanilla RNN. We also implemented the sample funciton for LSTM.

- Q3: Image Gradients: Saliency maps and Fooling Images 



- Q4: Image Generation: Classes, Inversion, DeepDream 
