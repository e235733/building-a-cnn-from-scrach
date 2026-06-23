from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()
from common.functions import *
from common.util import *

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = xp.tanh(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1 - self.out**2)
        return dx


class Relu:
    def __init__(self):
        self.mask = None
        self.out = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        self.out = out
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class LeakyRelu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mask = None
        self.out = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] *= self.alpha
        self.out = out
        return out

    def backward(self, dout):
        dout[self.mask] *= self.alpha
        dx = dout
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[xp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.out = None
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = xp.dot(self.x, self.W) + self.b
        self.out = out
        return out

    def backward(self, dout):
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        
        return dx.reshape(*self.original_x_shape)


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        self.out = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = xp.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        self.out = out

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = xp.sum(dout, axis=0)
        self.dW = xp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = xp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        self.out = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = xp.argmax(col, axis=1)
        out = xp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        self.out = out

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = xp.zeros((dout.size, pool_size))
        dmax[xp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class GlobalAveragePooling:
    def __init__(self):
        self.x = None
        self.out = None

    def forward(self, x):
        self.x = x
        # N, C, H, W = x.shape
        out = xp.mean(x, axis=(2, 3))  # (N, C)
        self.out = out
        return out

    def backward(self, dout):
        N, C = dout.shape
        H, W = self.x.shape[2], self.x.shape[3]
        dx = dout[:, :, None, None] / (H * W)  # (N, C, 1, 1)
        dx = xp.tile(dx, (1, 1, H, W))  # (N, C, H, W)
        return dx

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, eps=1e-5):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.eps = eps

        # テスト時に使用する平均と分散
        self.running_mean = None
        self.running_var = None
        
        # 逆伝播時に使用する中間値
        self.batch_size = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, is_training=True):
        # 4次元入力(N,C,H,W)を2次元に変換 
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1) # NHWCに入れ替え
            x = x.reshape(N*H*W, C)
            out = self.calc_forward(x, is_training)
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2) # NCHWに戻す
        elif x.ndim == 2:
            out = self.calc_forward(x, is_training)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        return out
    
    def calc_forward(self, x, is_training):
        # 初回のみ running_mean, running_var を初期化
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = x.mean(axis=0)
            self.running_var = x.var(axis=0)

        if is_training:
            # バッチ統計量の計算
            mu = x.mean(axis=0)
            xc = x - mu
            var = xp.mean(xc**2, axis=0)
            std = xp.sqrt(var + self.eps)
            xn = xc / std
            
            # 逆伝播用に保存
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            
            # 実行時平均・分散の更新（指数移動平均）
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            # テスト時：保存された平均・分散を使用
            xc = x - self.running_mean
            std = xp.sqrt(self.running_var + self.eps)
            xn = xc / std
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1)
            dout = dout.reshape(N*H*W, C)
            dx = self.calc_backward(dout)
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        elif dout.ndim == 2:
            dx = self.calc_backward(dout)
        else:
            raise ValueError(f"Unsupported dout shape: {dout.shape}")
        return dx

    def calc_backward(self, dout):
        # パラメータ勾配の計算
        dbeta = xp.sum(dout, axis=0)
        dgamma = xp.sum(self.xn * dout, axis=0)
        
        # 入力勾配の計算（数値的に安定した一括計算式）
        dx = (self.gamma / (self.batch_size * self.std)) * (
            self.batch_size * dout - xp.sum(dout, axis=0) - 
            self.xn * xp.sum(dout * self.xn, axis=0)
        )

        # パラメータ勾配を保存
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
    
class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, is_training=True):
        if is_training:
            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class ResidualBlock:
    """Pre-activation残差ブロック(スキップ接続を含む)
    
    Pre-activationアーキテクチャを採用：
    BatchNorm → ReLU → Conv → BatchNorm → ReLU → Conv → (+) → 出力
    
    従来のPost-activationより勾配が安定し、より深いネットワークに対応。
    チャネル数やサイズが異なる場合は1x1 Convで調整する。
    """
    def __init__(self, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, stride=1, pad=1,
                 use_1x1_conv=False, W_1x1=None, b_1x1=None,
                 gamma_1x1=None, beta_1x1=None):
        """
        Args:
            gamma1, beta1: 第1のBatchNorm層のパラメータ
            gamma2, beta2: 第2のBatchNorm層のパラメータ
            W1, b1: 第1の畳み込み層の重みとバイアス
            W2, b2: 第2の畳み込み層の重みとバイアス
            stride: ストライド (第1層に適用)
            pad: パディング
            use_1x1_conv: ショートカット接続用に1x1 Convを使うかどうか
            W_1x1, b_1x1: ショートカット用の1x1 Conv の重みとバイアス
            gamma_1x1, beta_1x1: ショートカット用のBatchNorm層のパラメータ
        """
        # Pre-activationではReluが先に来る
        self.bn1 = BatchNormalization(gamma1, beta1)
        self.relu1 = Relu()
        self.conv1 = Convolution(W1, b1, stride=stride, pad=pad)
        self.bn2 = BatchNormalization(gamma2, beta2)
        self.relu2 = Relu()
        self.conv2 = Convolution(W2, b2, stride=1, pad=pad)
        
        self.use_1x1_conv = use_1x1_conv
        if use_1x1_conv:
            self.bn_1x1 = BatchNormalization(gamma_1x1, beta_1x1)
            self.conv_1x1 = Convolution(W_1x1, b_1x1, stride=stride, pad=0)
        
        self.x = None
        self.out = None
        
    def forward(self, x, is_training=True):
        """順伝播(Pre-activation)
        
        流れ: BatchNorm → ReLU → Conv1 → BatchNorm → ReLU → Conv2 → (+) 
        
        Args:
            x: 入力データ (N, C, H, W)
            is_training: 学習モードかどうか
            
        Returns:
            出力データ (N, C', H', W')
        """
        self.x = x
        
        # メイン路: BatchNorm → ReLU → Conv → BatchNorm → ReLU → Conv
        h = self.bn1.forward(x, is_training)
        h = self.relu1.forward(h)
        h = self.conv1.forward(h)
        h = self.bn2.forward(h, is_training)
        h = self.relu2.forward(h)
        h = self.conv2.forward(h)
        
        # ショートカット接続
        if self.use_1x1_conv:
            shortcut = self.bn_1x1.forward(x, is_training)
            shortcut = self.conv_1x1.forward(shortcut)
        else:
            shortcut = x
        
        # 加算（この後の出力が次のブロックの入力になり、
        # 次のブロックでReLUが適用される）
        out = h + shortcut
        self.out = out
        
        return out
    
    def backward(self, dout):
        """逆伝播
        
        Args:
            dout: 出力層からの勾配
            
        Returns:
            入力層への勾配
        """
        # 加算の逆伝播（勾配を2つに分岐）
        dh = dout
        dshortcut = dout
        
        # メイン路の逆伝播
        dh = self.conv2.backward(dh)
        dh = self.relu2.backward(dh)
        dh = self.bn2.backward(dh)
        dh = self.conv1.backward(dh)
        dh = self.relu1.backward(dh)
        dh = self.bn1.backward(dh)

        # ショートカット接続の逆伝播
        if self.use_1x1_conv:
            dshortcut = self.conv_1x1.backward(dshortcut)
            dshortcut = self.bn_1x1.backward(dshortcut)
        
        # 入力層への勾配
        dx = dh + dshortcut
        
        return dx
    
    def get_params(self):
        """パラメータを辞書で返す"""
        params = {}
        params['W1'] = self.conv1.W
        params['b1'] = self.conv1.b
        params['W2'] = self.conv2.W
        params['b2'] = self.conv2.b
        params['gamma1'] = self.bn1.gamma
        params['beta1'] = self.bn1.beta
        params['gamma2'] = self.bn2.gamma
        params['beta2'] = self.bn2.beta
        
        if self.use_1x1_conv:
            params['W_1x1'] = self.conv_1x1.W
            params['b_1x1'] = self.conv_1x1.b
            params['gamma_1x1'] = self.bn_1x1.gamma
            params['beta_1x1'] = self.bn_1x1.beta
            
        return params
    
    def get_grads(self):
        """勾配を辞書で返す"""
        grads = {}
        grads['W1'] = self.conv1.dW
        grads['b1'] = self.conv1.db
        grads['W2'] = self.conv2.dW
        grads['b2'] = self.conv2.db
        grads['gamma1'] = self.bn1.dgamma
        grads['beta1'] = self.bn1.dbeta
        grads['gamma2'] = self.bn2.dgamma
        grads['beta2'] = self.bn2.dbeta

        if self.use_1x1_conv:
            grads['W_1x1'] = self.conv_1x1.dW
            grads['b_1x1'] = self.conv_1x1.db
            grads['gamma_1x1'] = self.bn_1x1.dgamma
            grads['beta_1x1'] = self.bn_1x1.dbeta

        return grads