import numpy as np
from layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU, Softmax


def cross_entropy_loss(probs, y_true):
    eps = 1e-9  # avoid log(0)
    return -np.mean(np.sum(y_true * np.log(probs + eps), axis=1))


class ConvNet:
    """
    Simple CNN: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> ReLU -> FC -> Softmax

    Architecture:
        input:  (N, 1, 28, 28)
        conv1:  (N, 8, 28, 28)  [3x3, padding=1]
        pool1:  (N, 8, 14, 14)
        conv2:  (N, 16, 14, 14) [3x3, padding=1]
        pool2:  (N, 16, 7, 7)
        flatten:(N, 784)
        fc1:    (N, 128)
        fc2:    (N, 10)
        softmax:(N, 10)
    """

    def __init__(self):
        self.layers = [
            Conv2D(1, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Conv2D(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(16 * 7 * 7, 128),
            ReLU(),
            Dense(128, 10),
            Softmax(),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true):
        # softmax.backward takes labels directly, not upstream grad
        grad = self.layers[-1].backward(y_true)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'W') and layer.dW is not None:
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
