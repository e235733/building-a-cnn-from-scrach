from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()


def identity_function(x):
    return x


def step_function(x):
    return xp.array(x > 0, dtype=int)


def sigmoid(x):
    x_clipped = xp.clip(x, -250, 250)
    return 1 / (1 + xp.exp(-x_clipped))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def tanh(x):
    return xp.tanh(x)


def tanh_grad(x):
    return 1 - xp.tanh(x) ** 2


def relu(x):
    return xp.maximum(0, x)


def relu_grad(x):
    grad = xp.zeros_like(x)
    grad[x>=0] = 1
    return grad


def leaky_relu(x, alpha):
    return xp.maximum(x, x*alpha)


def leaky_relu_grad(x, alpha):
    grad = xp.ones_like(x)
    grad[x<0] = alpha
    return grad


def softmax(x):
    x = x - xp.max(x, axis=-1, keepdims=True)
    return xp.exp(x) / xp.sum(xp.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * xp.sum((y-t)**2)


def cross_entropy_error(p, y):
    if p.ndim == 1:
        y = y.reshape(1, y.size)
        p = p.reshape(1, p.size)

    if y.size == p.size:
        y = y.argmax(axis=1)

    batch_size = y.shape[0]
    return -xp.sum(xp.log(p[xp.arange(batch_size), y] + 1e-7)) / batch_size


def softmax_loss(A, y):
    p = softmax(A)
    return cross_entropy_error(p, y)
