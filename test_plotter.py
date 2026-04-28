
import sys
import os
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

# プロジェクトルートのパスを通す
sys.path.append(os.getcwd())

from plotter import Plotter
from common.layers import Convolution, Affine, Relu

# 1. テスト用のダミーモデル
class DummyModel:
    def __init__(self):
        self.layers = OrderedDict()
        # Convolution層 (FN=16, C=1, FH=3, FW=3)
        self.layers['Conv1'] = Convolution(np.random.randn(16, 1, 3, 3), np.zeros(16))
        self.layers['Relu1'] = Relu()
        # Affine層
        self.layers['Affine1'] = Affine(np.random.randn(16*26*26, 10), np.zeros(10))
        
        # 順伝播のシミュレーション（統計情報用）
        x = np.random.randn(1, 1, 28, 28)
        for layer in self.layers.values():
            x = layer.forward(x)
        
        # 逆伝播のシミュレーション（勾配情報用）
        # 各レイヤーに dW を持たせる
        self.layers['Conv1'].dW = np.random.randn(16, 1, 3, 3)
        self.layers['Affine1'].dW = np.random.randn(16*26*26, 10)

    def predict(self, x):
        # 決定境界描画テスト用
        return np.random.rand(len(x), 10)

# 2. テスト用のダミーTrainer
class DummyTrainer:
    def __init__(self, model):
        self.model = model
        self.train_loss_list = [2.5, 2.0, 1.5, 1.2, 0.8]
        self.train_acc_list = [0.1, 0.3, 0.5, 0.7, 0.8]
        self.test_acc_list = [0.1, 0.25, 0.45, 0.65, 0.75]

def test_plotter_functionality():
    print("Testing Plotter with new layer structure and Trainer interface...")
    
    # ダミーデータの作成 (2次元入力としてシミュレート)
    X = np.random.randn(100, 2)
    Y = np.zeros((100, 10))
    for i in range(100): Y[i, i%10] = 1
    
    model = DummyModel()
    trainer = DummyTrainer(model)
    
    # Plotterの初期化 (2次元データ、詳細モード)
    plotter = Plotter(interval=1, X=X, Y=Y, is_detail_mode=True)
    
    try:
        # 描画の実行
        plotter.show(trainer)
        print("Success: Plotter.show() executed without errors.")
        
        # 評価画面のテスト
        X_test = np.random.randn(20, 2)
        Y_test = np.zeros((20, 10))
        plotter.show_evaluation(model, X_test, Y_test)
        print("Success: Plotter.show_evaluation() executed without errors.")
        
        plt.close('all')
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plotter_functionality()
