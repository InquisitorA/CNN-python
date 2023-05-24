# CNN-python
Training a simple CNN with single convolution, pooling and dense layer on the MNIST dataset

MNIST: train images -> 60000, 28x28x1
       test images  -> 10000, 28x28x1

CNN: convolution -> batch size = 32, filter = 3x3
     pooling     -> maxpooling, poolsize = 2x2
     dense       -> relu, 64 neurons
     dense       -> softmax, 10 neurons
     
RUN: Install tensorflow in the environment and run the command "python cnn.py"
