import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    x_clipped = np.clip(x, -250, 250)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad


def leaky_relu(x, alpha):
    return np.maximum(x, x*alpha)


def leaky_relu_grad(x, alpha):
    grad = np.ones_like(x)
    grad[x<0] = alpha
    return grad


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(p, y):
    if p.ndim == 1:
        y = y.reshape(1, y.size)
        p = p.reshape(1, p.size)

    if y.size == p.size:
        y = y.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(p[np.arange(batch_size), y] + 1e-7)) / batch_size


def softmax_loss(A, y):
    p = softmax(A)
    return cross_entropy_error(p, y)