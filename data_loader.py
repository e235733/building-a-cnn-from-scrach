from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()

class DataLoader:
    def __init__(self, X, Y, batch_size=1000, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]

        self.indices = xp.arange(self.n_samples)
        self.reset()

    # 再シャッフルしてリセット
    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            xp.random.shuffle(self.indices)

    
    def __len__(self):
        return int(xp.ceil(self.n_samples / self.batch_size))
    
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
        
        # データの切り出し
        end_idx = min(self.current_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.current_idx:end_idx]

        batch_X = self.X[batch_indices]
        batch_Y = self.Y[batch_indices]


        self.current_idx = end_idx

        return batch_X, batch_Y
    
class DataNormalizer:
    def __init__(self, data):
        self.mean = xp.mean(data, axis=0)
        self.std = xp.std(data, axis=0) + 1e-8

    # 全体の統計量を使った正規化
    def normalize(self, data):
        return (data - self.mean) / self.std


if __name__ == "__main__":
    from data_loader import DataLoader
    X = xp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 0]])
    Y = xp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    data_loader = DataLoader(X, Y, 2, True)

    for batch_X, batch_Y in data_loader:
        print(batch_X, batch_Y)
