from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()

class SGD:
    def __init__(self, lr=0.01, weight_decay=0):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, params, grads):
        for key in params.keys():
            if key in grads:
                if self.weight_decay > 0 and 'W' in key:
                    grads[key] += self.weight_decay * params[key]
                params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = xp.zeros_like(val)
                
        for key in params.keys():
            if key in grads:
                if self.weight_decay > 0 and 'W' in key:
                    grads[key] += self.weight_decay * params[key]
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]

class RMSProp:
    def __init__(self, lr=0.01, alpha=0.9, weight_decay=0):
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = xp.zeros_like(val)

        for key in params.keys():
            if key in grads:
                if self.weight_decay > 0 and 'W' in key:
                    grads[key] += self.weight_decay * params[key]
                self.h[key] = self.alpha * self.h[key] + (1.0 - self.alpha) * grads[key] ** 2
                params[key] -= self.lr * grads[key] / (xp.sqrt(self.h[key]) + 1e-7)
