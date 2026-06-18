from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()
from common.layers import *
from collections import OrderedDict

# ネットワークを設計可能なモデル
class NN_Model:
    def __init__(self, input_dim=(1, 28, 28), 
                 layer_config=[['Conv', 30, 5, 2, 1], ['Relu'], ['Pool', 2, 2, 2], ['Affine', 100], ['Relu'], ['Affine', 10]], 
                 last_layer='Softmax', weight_init_type='he', use_batchnorm=False):

        self.params = {}
        self.layers = OrderedDict()
        self.layer_params = OrderedDict()  # layer_name -> [param_keys]を記録

        latest_dim = input_dim
        param_count = 0
        layer_count = {'Conv': 0, 'Pool': 0, 'Affine': 0, 'Relu': 0, 'LeakyRelu': 0, 'Dropout': 0, 'ResidualBlock': 0}
        if use_batchnorm:
            layer_count['BatchNorm'] = 0
        
        self.is_training = False
        self.use_batchnorm = use_batchnorm

        for _, l in enumerate(layer_config):
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
                    conv_layer_name = 'Conv'+layer_ord
                    self.layers[conv_layer_name] = Convolution(self.params['W'+param_ord], self.params['b'+param_ord], filter_stride, filter_pad)
                    self.layer_params[conv_layer_name] = ['W'+param_ord, 'b'+param_ord]

                    out_h = 1 + int((latest_dim[1] + 2*filter_pad - filter_size) / filter_stride)
                    out_w = 1 + int((latest_dim[2] + 2*filter_pad - filter_size) / filter_stride)
                    latest_dim=(filter_num, out_h, out_w)
                    
                    if self.use_batchnorm:
                        self.params['gamma'+param_ord] = xp.ones(filter_num)
                        self.params['beta'+param_ord] = xp.zeros(filter_num)

                        layer_count['BatchNorm'] += 1
                        layer_ord = str(layer_count['BatchNorm'])
                        bn_layer_name = 'BatchNorm'+layer_ord
                        self.layers[bn_layer_name] = BatchNormalization(self.params['gamma'+param_ord], self.params['beta'+param_ord])
                        self.layer_params[bn_layer_name] = ['gamma'+param_ord, 'beta'+param_ord]
                
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
                    affine_layer_name = 'Affine'+layer_ord
                    self.layers[affine_layer_name] = Affine(self.params['W'+param_ord], self.params['b'+param_ord])
                    self.layer_params[affine_layer_name] = ['W'+param_ord, 'b'+param_ord]

                    latest_dim = (output_size, )

                case 'Relu':
                    layer_count['Relu'] += 1
                    layer_ord = str(layer_count['Relu'])
                    self.layers['Relu'+layer_ord] = Relu()

                case 'LeakyRelu':
                    layer_count['LeakyRelu'] += 1
                    layer_ord = str(layer_count['LeakyRelu'])
                    self.layers['LeakyRelu'+layer_ord] = LeakyRelu()

                case 'Dropout':
                    layer_count['Dropout'] += 1
                    layer_ord = str(layer_count['Dropout'])
                    self.layers['Dropout'+layer_ord] = Dropout(0.5)

                case 'ResidualBlock':
                    # layer_config: ['ResidualBlock', out_channels, filter_size, stride, pad]
                    # または: ['ResidualBlock', out_channels, filter_size, stride, pad, use_1x1_conv]
                    out_channels = l[1]
                    filter_size = l[2]
                    stride = l[3] if len(l) > 3 else 1
                    pad = l[4] if len(l) > 4 else 1
                    use_1x1_conv_flag = l[5] if len(l) > 5 else (stride != 1 or out_channels != latest_dim[0])
                    
                    in_channels = latest_dim[0]
                    
                    # ノード数から初期値の標準偏差を計算
                    node_num = in_channels * filter_size * filter_size
                    weight_init_std = self._get_weight_init_std(weight_init_type, node_num)
                    
                    # メイン路の重み
                    param_count += 1
                    W1_key = 'W' + str(param_count)
                    self.params[W1_key] = weight_init_std * \
                        xp.random.randn(out_channels, in_channels, filter_size, filter_size)
                    
                    param_count += 1
                    b1_key = 'b' + str(param_count)
                    self.params[b1_key] = xp.zeros(out_channels)
                    
                    param_count += 1
                    W2_key = 'W' + str(param_count)
                    self.params[W2_key] = weight_init_std * \
                        xp.random.randn(out_channels, out_channels, filter_size, filter_size)
                    
                    param_count += 1
                    b2_key = 'b' + str(param_count)
                    self.params[b2_key] = xp.zeros(out_channels)
                    
                    # 1x1 Conv の重み（必要な場合）
                    if use_1x1_conv_flag:
                        param_count += 1
                        W_1x1_key = 'W' + str(param_count)
                        self.params[W_1x1_key] = weight_init_std * \
                            xp.random.randn(out_channels, in_channels, 1, 1)
                        
                        param_count += 1
                        b_1x1_key = 'b' + str(param_count)
                        self.params[b_1x1_key] = xp.zeros(out_channels)
                        
                        res_block = ResidualBlock(
                            self.params[W1_key], self.params[b1_key],
                            self.params[W2_key], self.params[b2_key],
                            stride=stride, pad=pad,
                            use_1x1_conv=True,
                            W_1x1=self.params[W_1x1_key],
                            b_1x1=self.params[b_1x1_key]
                        )
                    else:
                        res_block = ResidualBlock(
                            self.params[W1_key], self.params[b1_key],
                            self.params[W2_key], self.params[b2_key],
                            stride=stride, pad=pad,
                            use_1x1_conv=False
                        )
                    
                    layer_count['ResidualBlock'] += 1
                    layer_ord = str(layer_count['ResidualBlock'])
                    res_block_name = 'ResidualBlock'+layer_ord
                    self.layers[res_block_name] = res_block
                    
                    # ResidualBlock のパラメータキーを記録
                    res_param_keys = [W1_key, b1_key, W2_key, b2_key]
                    if use_1x1_conv_flag:
                        res_param_keys.extend([W_1x1_key, b_1x1_key])
                    self.layer_params[res_block_name] = res_param_keys
                    
                    # 出力サイズを計算
                    out_h = 1 + int((latest_dim[1] + 2*pad - filter_size) / stride)
                    out_w = 1 + int((latest_dim[2] + 2*pad - filter_size) / stride)
                    latest_dim = (out_channels, out_h, out_w)

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

    def _forward(self, x):
        for layer in self.layers.values():
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer.forward(x, self.is_training)
            else:
                x = layer.forward(x)
        return x

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
                out = self._forward(tx)
                
                if result is None:
                    # 出力形状の動的取得と初期化
                    result = xp.zeros((x.shape[0], *out.shape[1:]), dtype=out.dtype)
                result[i*batch_size:(i+1)*batch_size] = out
            return result
        else:
            return self._forward(x)

    def loss(self, x, t):
        """損失関数を求める
        x: 予測データ
        t: 正解ラベル
        """
        # 勾配計算に必要な中間状態を保存するため、バッチ分割せずに順伝播を行う
        self.is_training = True
        y = self.predict(x, batch_size=None)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=64):
        """正解率を求める
        x: 予測データ
        t: 正解ラベル
        batch_size: 正解率計算のバッチサイズ
        """
        self.is_training = False
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
        self.last_loss = self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 勾配の設定
        grads = {}
        
        for layer_name, layer in self.layers.items():
            # ResidualBlock層の勾配
            if layer_name.startswith('ResidualBlock') and hasattr(layer, 'get_grads'):
                res_grads = layer.get_grads()
                param_keys = self.layer_params[layer_name]
                
                # res_grads の順序は W1, b1, W2, b2, [W_1x1, b_1x1]
                grad_keys = list(res_grads.keys())
                for i, key in enumerate(grad_keys):
                    grads[param_keys[i]] = res_grads[key]
            
            # Convolution層の勾配
            elif layer_name.startswith('Conv') and hasattr(layer, 'dW'):
                param_keys = self.layer_params[layer_name]
                grads[param_keys[0]] = layer.dW
                grads[param_keys[1]] = layer.db
            
            # BatchNormalization層の勾配
            elif layer_name.startswith('BatchNorm') and hasattr(layer, 'dgamma'):
                param_keys = self.layer_params[layer_name]
                grads[param_keys[0]] = layer.dgamma
                grads[param_keys[1]] = layer.dbeta
            
            # Affine層の勾配
            elif layer_name.startswith('Affine') and hasattr(layer, 'dW'):
                param_keys = self.layer_params[layer_name]
                grads[param_keys[0]] = layer.dW
                grads[param_keys[1]] = layer.db

        return grads
