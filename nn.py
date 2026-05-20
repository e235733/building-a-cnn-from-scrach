from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()
from common.layers import *
from collections import OrderedDict

# ネットワークを設計可能なモデル
class NN_Model:
    def __init__(self, input_dim=(1, 28, 28), 
                 layer_config=[['Conv', 30, 5, 2, 1], ['Relu'], ['Pool', 2, 2, 2], ['Affine', 100], ['Relu'], ['Affine', 10]], 
                 last_layer='Softmax', weight_init_type='he'):

        self.params = {}
        self.layers = OrderedDict()

        latest_dim = input_dim
        param_count = 0
        layer_count = {'Conv': 0, 'Pool': 0, 'Affine': 0, 'Relu': 0, 'LeakyRelu': 0}
        
        for i, l in enumerate(layer_config):
            match l[0]:
                case 'Conv':
                    filter_num = l[1]
                    filter_size = l[2]
                    filter_pad  = l[3]
                    filter_stride = l[4]

                    # ノード数から初期値の標準偏差を計算
                    node_num = latest_dim[0] * filter_size * filter_size
                    weight_init_std = self._get_weight_init_std(weight_init_type, node_num)

                    param_count += 1
                    param_ord = str(param_count)

                    self.params['W'+param_ord] = weight_init_std * \
                    xp.random.randn(filter_num, latest_dim[0], filter_size, filter_size)

                    self.params['b'+param_ord] = xp.zeros(filter_num)

                    layer_count['Conv'] += 1
                    layer_ord = str(layer_count['Conv'])
                    self.layers['Conv'+layer_ord] = Convolution(self.params['W'+param_ord], self.params['b'+param_ord], filter_stride, filter_pad)

                    out_h = 1 + int((latest_dim[1] + 2*filter_pad - filter_size) / filter_stride)
                    out_w = 1 + int((latest_dim[2] + 2*filter_pad - filter_size) / filter_stride)
                    latest_dim=(filter_num, out_h, out_w)

                case 'Pool':
                    pool_h = l[1]
                    pool_w = l[2]
                    pool_stride = l[3]

                    layer_count['Pool'] += 1
                    layer_ord = str(layer_count['Pool'])
                    self.layers['Pool'+layer_ord] = Pooling(pool_h, pool_w, pool_stride)

                    out_h = int(1 + (latest_dim[1] - pool_h) / pool_stride)
                    out_w = int(1 + (latest_dim[2] - pool_w) / pool_stride)
                    latest_dim = (latest_dim[0], out_h, out_w)

                case 'Affine':
                    input_size = None
                    if (len(latest_dim) == 3):
                        input_size = latest_dim[0] * latest_dim[1] * latest_dim[2]
                    elif (len(latest_dim) == 1):
                        input_size = latest_dim[0]
                    output_size = l[1]

                    # ノード数から初期値の標準偏差を計算
                    weight_init_std = self._get_weight_init_std(weight_init_type, input_size)

                    param_count += 1
                    param_ord = str(param_count)

                    self.params['W'+param_ord] = weight_init_std * \
                    xp.random.randn(input_size, output_size)
                    self.params['b'+param_ord] = xp.zeros(output_size)

                    layer_count['Affine'] += 1
                    layer_ord = str(layer_count['Affine'])
                    self.layers['Affine'+layer_ord] = Affine(self.params['W'+param_ord], self.params['b'+param_ord])

                    latest_dim = (output_size, )

                case 'Relu':
                    layer_count['Relu'] += 1
                    layer_ord = str(layer_count['Relu'])
                    self.layers['Relu'+layer_ord] = Relu()

                case 'LeakyRelu':
                    layer_count['LeakyRelu'] += 1
                    layer_ord = str(layer_count['LeakyRelu'])
                    self.layers['LeakyRelu'+layer_ord] = LeakyRelu()

        match last_layer:
            case 'Softmax':
                self.last_layer = SoftmaxWithLoss()

    def _get_weight_init_std(self, weight_init_type, node_num):
        """重みの標準偏差を計算する"""
        if str(weight_init_type).lower() in ('relu', 'he', 'leakyrelu'):
            return xp.sqrt(2.0 / node_num)
        elif str(weight_init_type).lower() in ('sigmoid', 'xavier'):
            return xp.sqrt(1.0 / node_num)
        
        # 数値指定（固定値）の場合
        try:
            return float(weight_init_type)
        except (ValueError, TypeError):
            return 0.01

    def predict(self, x, batch_size=None):
        """順伝播による予測
        x: 入力データ
        """
        x = xp.asarray(x)
        
        # 巨大なデータの入力に対応するためバッチ処理で計算（OOM対策）
        # 学習時のforwardで中間状態が上書きされるのを防ぐため、明示的な指定がない場合は一括で処理する
        if batch_size is not None and x.shape[0] > batch_size:
            result = None
            for i in range(int(xp.ceil(x.shape[0] / batch_size))):
                tx = x[i*batch_size:(i+1)*batch_size]
                # 各バッチに対して順伝播
                out = tx
                for layer in self.layers.values():
                    out = layer.forward(out)
                
                if result is None:
                    # 出力形状の動的取得と初期化
                    result = xp.zeros((x.shape[0], *out.shape[1:]), dtype=out.dtype)
                result[i*batch_size:(i+1)*batch_size] = out
            return result
        else:
            for layer in self.layers.values():
                x = layer.forward(x)
            return x

    def loss(self, x, t):
        """損失関数を求める
        x: 予測データ
        t: 正解ラベル
        """
        # 勾配計算に必要な中間状態を保存するため、バッチ分割せずに順伝播を行う
        y = self.predict(x, batch_size=None)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=64):
        """正解率を求める
        x: 予測データ
        t: 正解ラベル
        batch_size: 正解率計算のバッチサイズ
        """
        if t.ndim != 1:
            t = xp.argmax(t, axis=1)

        y = self.predict(x, batch_size=batch_size)
        y = xp.argmax(y, axis=1)
        num_correct = xp.sum(y == t)
        
        return num_correct / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める
        x : 入力データ
        t : 教師ラベル
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
        param_count = 1
        for layer in self.layers.values():
            if hasattr(layer, 'dW'):
                grads['W' + str(param_count)] = layer.dW
                grads['b' + str(param_count)] = layer.db
                param_count += 1

        return grads
