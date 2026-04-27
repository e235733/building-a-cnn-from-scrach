import numpy as np
from common.layers import *
from collections import OrderedDict

# 畳み込みニューラルネットワークのモデルクラス
class CNN_Model:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, 
                 hidden_size=100, output_size=10, weight_init_std=0.01, eta=0.01):

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.eta = eta # 学習率

        self.conv_W = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.conv_b = np.zeros(filter_num)
        self.affine1_W = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.affine1_b = np.zeros(hidden_size)
        self.affine2_W = weight_init_std * np.random.randn(hidden_size, output_size)
        self.affine1_b = np.zeros(output_size)

        # グラフ作成用の損失記録
        self.loss_history = []

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['W3'])

        self.last_layer = SoftmaxWithLoss()
        

    def update_parameters(self):
        # パラメータの更新
        for i in range(self.depth + 1):
            self.V_W[i] = self.alpha * self.V_W[i] - self.eta * self.dW[i]
            self.V_b[i] = self.alpha * self.V_b[i] - self.eta * self.db[i]
            self.W[i] += self.V_W[i]
            self.b[i] += self.V_b[i]


    def predict(self, x):
        """順伝播による予測
        x: 入力データ
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    def loss(self, x, t):
        """損失関数を求める
        x: 予測データ
        t: 正解ラベル
        """
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)
        self.loss_history.append(loss)
        return loss


    def accuracy(self, x, t, batch_size=100):
        """正解率を求める
        x: 予測データ
        t: 正解ラベル
        batch_size: 正解率計算のバッチサイズ
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        num_correct = 0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            num_correct += np.sum(y == tt) 
        return num_correct / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    