CNN from Scratch using NumPy

Built a convolutional neural network from scratch using only NumPy - no PyTorch or TensorFlow for the math. Made this to understand how backprop actually works through conv layers.

What's implemented:
- Conv2D (forward and backward pass)
- MaxPooling2D
- Fully connected layer
- ReLU and Softmax activations
- Cross-entropy loss
- SGD optimizer

Architecture:
Input (1x28x28) -> Conv(8 filters) -> ReLU -> MaxPool
               -> Conv(16 filters) -> ReLU -> MaxPool
               -> Flatten -> FC(128) -> ReLU -> FC(10) -> Softmax

How to run:
pip install numpy scikit-learn
python train.py

Trains on a 2000-sample subset of MNIST by default since full training is slow without GPU.
You can change num_train in train.py to use more data.

Files:
- layers.py - individual layer implementations with forward and backward pass
- network.py - model class that puts layers together
- train.py - data loading and training loop

Note: training on full MNIST takes around 30 minutes for 5 epochs because the conv operations use nested loops instead of vectorized im2col. That's intentional - easier to follow the math that way.
