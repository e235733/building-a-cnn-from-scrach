import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


class Convolution:
    def __init__(self, F:np.ndarray, b):
        self.F = F
        self.col_F = F.reshape(self.F.shape[0], -1).T
        self.b = b

        self.dcol_F = None
        self.db = None


    def forward(self, A):
        self.A_shape = A.shape
        _, _, f_h, f_w = self.F.shape
        self.col_A = im2col(A, f_h, f_w)
        conv = np.dot(self.col_A, self.col_F) + self.b
        return conv
        
    def backward(self, dout):
        self.dcol_F = self.col_A.T @ dout
        self.db = np.sum(dout, axis=0)
        dcol_A = dout @ self.col_F.T
        dA = dcol_A.reshape(*self.A_shape)
        return dA


if __name__ == "__main__":
    from toy_closs_dataset import ToyCrossDataset
    # 簡単なテストケース
    np.random.seed(0)
    
    data = ToyCrossDataset() # 入力データ
    x, y = data.load_data()
    W = np.random.rand(1, 1, 2, 2) # フィルタの重み
    b = np.random.rand(1) # フィルタのバイアス

    conv = Convolution(W, b)
    out = conv.forward(x)
    print("Convolution Forward Output:\n", out)

    dout = np.random.rand(*out.shape) # 上流からの勾配
    dx = conv.backward(dout)
    print("Convolution Backward dx:\n", dx)

    # pool = MaxPooling(pool_h=2, pool_w=2, stride=2)
    # out_pool = pool.forward(out)
    # print("MaxPooling Forward Output:\n", out_pool)

    # dout_pool = np.random.rand(*out_pool.shape) # 上流からの勾配
    # dx_pool = pool.backward(dout_pool)
    # print("MaxPooling Backward dx:\n", dx_pool)