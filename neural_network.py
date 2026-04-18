import numpy as np
import function as fn

# ニューラルネットワークのモデルクラス
class NeuralNetworkModel:
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
        self.W = []
        self.b = []
        
        rng = np.random.default_rng()
        
        layers = [self.input_dim] + self.hidden_layer + [self.output_dim]
        
        for i in range(len(layers) - 1):
            head = layers[i]
            tail = layers[i+1]
            
            # 活性化関数に基づいた初期化スケールを取得
            scale = self.act_fn.init_weight(head, tail)
            
            w = rng.standard_normal((head, tail)) * scale
            b = np.zeros(tail)
            
            self.W.append(w)
            self.b.append(b)

        # Momentum 用の速度 V をゼロ初期化
        self.V_w = [np.zeros_like(w) for w in self.W]
        self.V_b = [np.zeros_like(b) for b in self.b]
    
    def calc_forward_propagation(self, X):
        # 前向き伝播
        L = self.depth
        self.A = [X]
        for i in range(L):
            Z = self.A[i] @ self.W[i] + self.b[i]
            self.A.append(self.act_fn.value(Z))
        Z = self.A[L] @ self.W[L] + self.b[L]
        self.P = self.output_fn.value(Z)

    def append_grad(self, i, dz, threshold=5.0):
        # i 番目の dw, db を dz から計算
        dw = self.A[i].T @ dz + self.l2_lambda * self.W[i]
        dw_clipped = np.clip(dw, -threshold, threshold)
        db = np.sum(dz, axis=0)
        
        self.dW.append(dw_clipped)
        self.db.append(db)

    def calc_backward_propagation(self, Y):
        # 逆向き伝播
        self.dW = []
        self.db = []
        dz = self.output_fn.dLoss(self.P, Y)
        self.append_grad(self.depth, dz)
        for i in range(self.depth, 0, -1):
            da_prev = dz @ self.W[i].T
            dz = da_prev * self.act_fn.diff(self.A[i])
            self.append_grad(i-1, dz)
        self.dW.reverse()
        self.db.reverse()

    def update_parameters(self):
        # パラメータの更新
        for i in range(self.depth + 1):
            self.V_w[i] = self.alpha * self.V_w[i] - self.eta * self.dW[i]
            self.V_b[i] = self.alpha * self.V_b[i] - self.eta * self.db[i]
            self.W[i] += self.V_w[i]
            self.b[i] += self.V_b[i]

    def shift(self, X, Y):
        # 勾配を計算
        self.calc_forward_propagation(X)
        self.calc_backward_propagation(Y)
        # パラメータを更新
        self.update_parameters()

    def predict(self, X):
        A = X       
        for i in range(self.depth):
            Z = A @ self.W[i] + self.b[i]
            A = self.act_fn.value(Z)
        i = self.depth
        Z = A @ self.W[i] + self.b[i]
        return self.output_fn.value(Z)
    
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

    
if __name__ == "__main__":
    import function as fn

    # 簡易テスト
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # XOR っぽいラベル

    model = NeuralNetworkModel(
        input_dim=2,
        hidden_layer=[4, 4],
        output_dim=2,
        act_fn=fn.Sigmoid(),
        output_fn=fn.Softmax(),
        eta=0.1
    )

    print("Initial Loss:", model.loss(X, Y))
    for _ in range(100):
        model.shift(X, Y)
    print("Loss after 100 steps:", model.loss(X, Y))
    print("Predictions:\n", model.predict(X))
