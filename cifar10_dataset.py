import pickle
import os
import numpy as np
from common import config

xp = config.get_xp()

class Cifar10Dataset:
    def __init__(self, data_dir='cifar-10-batches-py', n_samples=None):
        self.data_dir = data_dir
        
        # 訓練データの読み込み
        x_train = []
        y_train = []
        
        try:
            for i in range(1, 6):
                file_path = os.path.join(self.data_dir, f'data_batch_{i}')
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                with open(file_path, 'rb') as f:
                    d = pickle.load(f, encoding='bytes')
                    x_train.append(d[b'data'])
                    y_train.extend(d[b'labels'])
            
            self.X_train = np.concatenate(x_train).astype(np.float32) / 255.0
            self.Y_train = np.identity(10)[y_train].astype(np.float32)
            
            # テストデータの読み込み
            test_file_path = os.path.join(self.data_dir, 'test_batch')
            if not os.path.exists(test_file_path):
                 raise FileNotFoundError(f"File not found: {test_file_path}")

            with open(test_file_path, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                self.X_test = d[b'data'].astype(np.float32) / 255.0
                self.Y_test = np.identity(10)[d[b'labels']].astype(np.float32)

            # メタデータの読み込み（クラス名）
            meta_file_path = os.path.join(self.data_dir, 'batches.meta')
            if os.path.exists(meta_file_path):
                with open(meta_file_path, 'rb') as f:
                    meta = pickle.load(f, encoding='bytes')
                    self.label_names = [name.decode('utf-8') for name in meta[b'label_names']]
            else:
                self.label_names = [str(i) for i in range(10)]

        except Exception as e:
            print(f"Error loading CIFAR-10 data: {e}")
            raise

        # GPUが有効な場合はデータを転送
        if config.GPU_ENABLE:
            self.X_train = xp.asarray(self.X_train)
            self.Y_train = xp.asarray(self.Y_train)
            self.X_test = xp.asarray(self.X_test)
            self.Y_test = xp.asarray(self.Y_test)

        # 指定されたサンプル数だけ抽出
        if n_samples:
            self.X_train = self.X_train[:n_samples]
            self.Y_train = self.Y_train[:n_samples]

        print(f"CIFAR-10 loaded: Train={len(self.X_train)}, Test={len(self.X_test)}")
