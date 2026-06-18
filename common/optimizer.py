from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            if key in grads:
                params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = xp.zeros_like(val)
                
        for key in params.keys():
            if key in grads:
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]

class RMSProp:
    def __init__(self, lr=0.01, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():                                
                self.h[key] = xp.zeros_like(val)
                
        for key in params.keys():
            if key in grads:
                self.h[key] = self.alpha * self.h[key] - (1.0 - self.lr) * grads[key] ** 2
                params[key] -= self.lr * grads[key] / (xp.sqrt(self.h[key]) + 1e-7)
