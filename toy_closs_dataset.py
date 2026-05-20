from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()

class ToyClossDataset:
    """トイデータセット（渦巻き状のデータ）を生成するクラス"""
    def __init__(self, num_classes=3, points_per_class=100):
        self.num_classes = num_classes
        self.points_per_class = points_per_class
        self.X, self.Y = self._generate_data()

    def _generate_data(self):
        N = self.points_per_class
        D = 2 # 2次元
        K = self.num_classes
        X = xp.zeros((N*K, D))
        Y = xp.zeros((N*K, K), dtype='uint8')
        
        for j in range(K):
            ix = range(N*j, N*(j+1))
            r = xp.linspace(0.0, 1, N) # 半径
            t = xp.linspace(j*4, (j+1)*4, N) + xp.random.randn(N)*0.2 # 角度
            X[ix] = xp.c_[r*xp.sin(t), r*xp.cos(t)]
            Y[ix, j] = 1
            
        return X, Y
