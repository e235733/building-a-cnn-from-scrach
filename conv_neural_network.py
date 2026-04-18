import numpy as np
import function as fn

# 畳み込みニューラルネットワークのモデルクラス
class ConvolutionalNeuralNetworkModel:
    def __init__(self, input_dim, hidden_layer, output_dim,
                act_fn:fn.ActivationFunction = fn.LeakyReLU(),
                output_fn:fn.OutputFunction = fn.Softmax(),
                eta=0.01, l2_lambda=0.005, alpha=0.9):
        
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.output_dim = output_dim
        self.depth = len(hidden_layer)

        self.act_fn = act_fn # 隠れ層の活性化関数
        self.output_fn = output_fn # 出力層の活性化関数

        self.eta = eta # 学習率
        self.l2_lambda = l2_lambda # L2正則化のペナルティ
        self.alpha = alpha # 慣性係数

        # パラメータの初期化
        self._initialize_parameters()

        # グラフ作成用の損失記録
        self.train_loss_history = [] 
        self.test_loss_history = []


    def _initialize_parameters(self):
        # 畳み込み層のパラメータの初期化
        self.conv_W = []
        self.conv_b = []
        
        rng = np.random.default_rng()
        
        # 畳み込み層の構成（例: 2層の畳み込み層）
        conv_layer_configs = [
            (3, 16, 3),  # (入力チャネル数, 出力チャネル数, カーネルサイズ)
            (16, 32, 3)
        ]
        
        for in_channels, out_channels, kernel_size in conv_layer_configs:
            scale = self.act_fn.init_wegit(in_channels * kernel_size * kernel_size, out_channels * kernel_size * kernel_size)
            w = rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size)) * scale
            b = np.zeros(out_channels)
            self.conv_W.append(w)
            self.conv_b.append(b)

        # 全結合層のパラメータの初期化
        layers = [32 * 7 * 7] + self.hidden_layer + [self.output_dim]
        
        for i in range(len(layers) - 1):
            head = layers[i]
            tail = layers[i+1]
            
            scale = self.act_fn.init_wegit(head, tail)
            
            w = rng.standard_normal((head, tail)) * scale
            b = np.zeros(tail)
            
            self.W.append(w)
            self.b.append(b)

        # Momentum 用の速度 V をゼロ初期化
        self.V_conv_W = [np.zeros_like(w) for w in self.conv_W]
        self.V_conv_b = [np.zeros_like(b) for b in self.conv_b]
        self.V_W = [np.zeros_like(w) for w in self.W]
        self.V_b = [np.zeros_like(b) for b in self.b]

    def convolution(self, A, w, b):
        # 畳み込み演算の実装（例: 単純な畳み込み）
        # ここでは簡略化のため、実際の畳み込み演算は省略
        return A  # ダミーの出力
    
    def pooling(self, A):
        # プーリング演算の実装（例: 最大プーリング）
        # ここでは簡略化のため、実際のプーリング演算は省略
        return A  # ダミーの出力

    def calc_forward_propagation(self, X: np.ndarray):
        # 前向き伝播の実装（例: 畳み込み層 → 活性化関数 → プーリング → 全結合層）
        # 畳み込み層の処理
        A = X
        for w, b in zip(self.conv_W, self.conv_b):
            A = self.convolution(A, w, b)
            A = self.act_fn.value(A)
            A = self.pooling(A)
        
        # 全結合層の処理
        A = A.reshape(A.shape[0], -1)  # フラット化
        for i in range(self.depth):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_fn.value(Z)
        Z = A @ self.W[self.depth] + self.b[self.depth]
        self.P = self.output_fn.value(Z)

    def calc_backward_propagation(self, Y):
        # 逆向き伝播の実装（例: 全結合層 → プーリング → 畳み込み層）
        # ここでは簡略化のため、実際の逆伝播演算は省略
        pass

    def update_parameters(self):
        # パラメータの更新
        for i in range(self.depth + 1):
            self.V_W[i] = self.alpha * self.V_W[i] - self.eta * self.dW[i]
            self.V_b[i] = self.alpha * self.V_b[i] - self.eta * self.db[i]
            self.W[i] += self.V_W[i]
            self.b[i] += self.V_b[i]

    def predict(self, X):
        # 予測の実装（例: 前向き伝播を通じてクラス確率を出力）
        A = X
        for w, b in zip(self.conv_W, self.conv_b):
            A = self.act_fn.value(self.convolution(A, w, b))
            A = self.pooling(A)
        A = A.reshape(A.shape[0], -1)  # フラット化
        for i in range(self.depth):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_fn.value(Z)
        Z = A @ self.W[self.depth] + self.b[self.depth]
        return self.output_fn.value(Z)
    
    def shift(self, X, Y):
        # 勾配を計算
        self.calc_forward_propagation(X)
        self.calc_backward_propagation(Y)
        # パラメータを更新
        self.update_parameters()

    def loss(self, X, Y):
        P = self.predict(X)
        base_loss = self.output_fn.Loss(P, Y)
        
        l2_penalty = 0.0
        for w in self.W:
            l2_penalty += np.sum(w ** 2) 
        l2_penalty *= (self.l2_lambda / 2)

        return l2_penalty + base_loss

    def log_train_loss(self, X_train, Y_train):
        train_loss = self.loss(X_train, Y_train)
        self.train_loss_history.append(train_loss)
        return train_loss

    def log_test_loss(self, X_test, Y_test):
        test_loss = self.loss(X_test, Y_test)
        self.test_loss_history.append(test_loss)
        return test_loss

    def evaluate_accuracy(self, X, Y: np.ndarray):
        # 予測と正解を比較して精度を計算
        predicted_classes = np.argmax(self.predict(X), axis=1)
        Y_labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        accuracy = np.mean(predicted_classes == Y_labels)
        return accuracy