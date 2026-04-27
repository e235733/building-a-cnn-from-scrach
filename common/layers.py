import numpy as np
from functions import *
from util import im2col, Momentum

class Sigmoid:
    def forward(self, A):
        self.out = sigmoid(A)
        return self.out
    
    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out
    
class Tanh:
    def forward(self, A):
        self.out = np.tanh(A)
        return self.out
    
    def backward(self, dout):
        return dout * (1 - self.out**2)
    
class ReLU:
    def forward(self, A):
        self.grad = relu_grad
        return np.maximum(0, x)
    
    def backward(self, dout):
        return dout * self.grad

class LeakyReLU:
    def forward(self, A):
        self.grad = leaky_relu_grad(A)
        return leaky_relu(A)
    
    def backward(self, dout):
        return dout * self.grad
    
class SoftmaxWithLoss:
    def forward(self, x, y):
        self.y = y
        self.p = softmax(x)
        self.loss = cross_entropy_error(self.p, self.y)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        if self.y.size == self.p.size: # 教師データがone-hot-vectorの場合
            dloss = (self.p - self.y) / batch_size
        else:
            dloss = self.p.copy()
            dloss[np.arange(batch_size), self.y] -= 1
            dloss = dloss / batch_size
        return dloss

class Affine:
    def __init__(self, W, b, eta = 0.01):
        self.W =W
        self.b = b
        self.momentum = Momentum(W, b, eta)
        
        self.col_A = None
        self.A_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, A):
        # テンソル対応
        self.A_shape = A.shape
        self.col_A = A.reshape(A.shape[0], -1)
        out = np.dot(self.col_A, self.W) + self.b

        return out
    
    def update_params(self):
        shift_W, shift_b = self.momentum.velocities(self.dW, self.db)
        self.W += shift_W
        self.b += shift_b

    def backward(self, dout):
        dA = np.dot(dout, self.W.T)
        self.dW = np.dot(self.col_A.T, dout)
        self.db = np.sum(dout, axis=0)
        self.update_params()
        return dA.reshape(*self.A_shape) # 入力データの形状に戻す（テンソル対応）

class Convolution:
    def __init__(self, F:np.ndarray, b, eta = 0.01):
        self.F = F
        self.col_F = F.reshape(self.F.shape[0], -1).T
        self.b = b
        self.momentum = Momentum(self.col_F, b, eta)

        self.dcol_F = None
        self.db = None

    def forward(self, A):
        self.A_shape = A.shape
        _, _, f_h, f_w = self.F.shape
        self.col_A = im2col(A, f_h, f_w)
        conv = np.dot(self.col_A, self.col_F) + self.b
        return conv
    
    def update_params(self):
        shift_F, shift_b = self.momentum.velocities(self.dcol_F, self.db)
        self.F += shift_F
        self.b += shift_b
        
    def backward(self, dout):
        self.dcol_F = self.col_A.T @ dout
        self.db = np.sum(dout, axis=0)
        dcol_A = dout @ self.col_F.T
        dA = dcol_A.reshape(*self.A_shape)
        self.update_params()
        return dA


if __name__ == "__main__":
    from toy_closs_dataset import ToyCrossDataset
    # 簡単なテストケース
    np.random.seed(0)
    
    data = ToyCrossDataset() # 入力データ
    x, y = data.load_data()
    W = np.random.rand(1, 1, 2, 2) # フィルタの重み
    b = np.random.rand(1) # フィルタのバイアス

    conv = Convolution(W, b)
    out = conv.forward(x)
    print("Convolution Forward Output:\n", out)

    dout = np.random.rand(*out.shape) # 上流からの勾配
    dx = conv.backward(dout)
    print("Convolution Backward dx:\n", dx)

    # pool = MaxPooling(pool_h=2, pool_w=2, stride=2)
    # out_pool = pool.forward(out)
    # print("MaxPooling Forward Output:\n", out_pool)

    # dout_pool = np.random.rand(*out_pool.shape) # 上流からの勾配
    # dx_pool = pool.backward(dout_pool)
    # print("MaxPooling Backward dx:\n", dx_pool)