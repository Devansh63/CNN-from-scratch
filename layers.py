import warnings
import numpy as np

# suppress spurious BLAS subnormal warnings on Apple Silicon
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")


class Conv2D:
    """2D convolution layer with forward and backward pass."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization - works better than random for deep nets
        fan_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels)
        self.dW = None
        self.db = None

    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        self.x_input = x
        batch, C, H, W = x.shape
        k = self.ksize
        s = self.stride
        p = self.padding

        if p > 0:
            x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        self.x_pad = x

        out_H = (H + 2 * p - k) // s + 1
        out_W = (W + 2 * p - k) // s + 1
        out = np.zeros((batch, self.out_ch, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                hs, ws = i * s, j * s
                patch = x[:, :, hs:hs + k, ws:ws + k]  # (batch, C, k, k)
                # sum over in_channels and spatial dims
                out[:, :, i, j] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b

        return out

    def backward(self, d_out):
        # d_out: (batch, out_ch, out_H, out_W)
        batch, C, H, W = self.x_input.shape
        _, _, out_H, out_W = d_out.shape
        k = self.ksize
        s = self.stride

        self.dW = np.zeros_like(self.W)
        self.db = d_out.sum(axis=(0, 2, 3))
        dx_pad = np.zeros_like(self.x_pad)

        for i in range(out_H):
            for j in range(out_W):
                hs, ws = i * s, j * s
                patch = self.x_pad[:, :, hs:hs + k, ws:ws + k]  # (batch, C, k, k)
                d_ij = d_out[:, :, i, j]  # (batch, out_ch)

                # accumulate weight gradients
                self.dW += np.tensordot(d_ij, patch, axes=([0], [0]))
                # distribute gradient back to input
                dx_pad[:, :, hs:hs + k, ws:ws + k] += np.tensordot(d_ij, self.W, axes=([1], [0]))

        if self.padding > 0:
            p = self.padding
            return dx_pad[:, :, p:-p, p:-p]
        return dx_pad


class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):
        self.x = x
        batch, C, H, W = x.shape
        p = self.pool_size
        out_H, out_W = H // p, W // p

        out = np.zeros((batch, C, out_H, out_W))
        for i in range(out_H):
            for j in range(out_W):
                out[:, :, i, j] = np.max(x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p], axis=(2, 3))
        return out

    def backward(self, d_out):
        batch, C, H, W = self.x.shape
        p = self.pool_size
        out_H, out_W = H // p, W // p
        dx = np.zeros_like(self.x)

        for i in range(out_H):
            for j in range(out_W):
                patch = self.x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p]
                max_val = np.max(patch, axis=(2, 3), keepdims=True)
                # only pass gradient through the max element
                mask = (patch == max_val).astype(float)
                mask /= mask.sum(axis=(2, 3), keepdims=True)  # handle ties
                dx[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p] += d_out[:, :, i, j][:, :, None, None] * mask

        return dx


class Flatten:
    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.original_shape)


class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        # grad: upstream gradient (batch, output_size)
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Softmax:
    """Softmax output layer. backward() expects one-hot labels and returns
    the combined gradient of cross-entropy loss + softmax."""

    def forward(self, x):
        # subtract max for numerical stability
        shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.probs = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, y_true):
        # gradient of CE loss w.r.t. pre-softmax input simplifies to this
        batch_size = y_true.shape[0]
        return (self.probs - y_true) / batch_size
