from sklearn.datasets import fetch_openml
import numpy as np

class MnistDataset:
    def __init__(self, n_samples=2000):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)

        X = mnist.data[:n_samples].astype(np.float32) / 255.0
        Y = np.identity(10)[mnist.target[:n_samples].astype(np.int32)]

        self.X_train = X[:60000]
        self.Y_train = Y[:60000]

        self.X_test = X[60000:]
        self.Y_test = Y[60000:]