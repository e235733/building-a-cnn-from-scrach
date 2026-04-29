import os
from sklearn.datasets import fetch_openml
import numpy as np

class MnistDataset:
    def __init__(self, n_samples=2000, cache_file='mnist.npz'):
        if os.path.exists(cache_file):
            print(f"Loading MNIST from {cache_file}...")
            data = np.load(cache_file)
            X_all = data['X']
            Y_all = data['Y']
        else:
            print("Downloading MNIST from OpenML (this may take a while)...")
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)

            # データの正規化とOne-hotエンコーディング
            X_all = mnist.data.astype(np.float32) / 255.0
            Y_all = np.identity(10)[mnist.target.astype(np.int32)]

            print(f"Saving to {cache_file}...")
            np.savez_compressed(cache_file, X=X_all, Y=Y_all)

        # 指定されたサンプル数だけ抽出
        X = X_all[:n_samples]
        Y = Y_all[:n_samples]

        # 訓練データとテストデータの分割 (標準は60000:10000)
        # サンプル数が少ない場合は 9:1 の比率で分割
        split_idx = min(60000, int(len(X) * 0.9))

        self.X_train = X[:split_idx]
        self.Y_train = Y[:split_idx]

        self.X_test = X[split_idx:]
        self.Y_test = Y[split_idx:]

        print(f"Dataset loaded: Train={len(self.X_train)}, Test={len(self.X_test)}")