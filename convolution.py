import numpy as np

# im2col関数は、入力データを畳み込み演算用の行列に変換する関数です。
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape # 入力データの形状を取得
    out_h = (H + 2*pad - filter_h) // stride + 1 # 出力の高さを計算
    out_w = (W + 2*pad - filter_w) // stride + 1 # 出力の幅を計算

    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant') # 入力データにパディングを追加
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) # 出力用の行列を初期化

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

# col2im関数は、im2colで変換された行列を元の画像形式に戻すための関数です。
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Convolution:
    def __init__(self, W, b, stride=1, pad=None):
        self.W = W # フィルタの重み
        self.b = b # フィルタのバイアス
        self.stride = stride # 畳み込みのストライド
        if pad is None:
            self.pad = (W.shape[2] - 1) // 2 # カーネルサイズに基づいて自動的にパディングを計算
        else:
            self.pad = pad # 畳み込みのパディング

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        dW = np.dot(im2col(self.x, FH, FW, self.stride, self.pad).T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        db = np.sum(dout, axis=0)

        dcol = np.dot(dout, self.W.reshape(FN, -1))
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx, dW, db
    
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h # プーリングの高さ
        self.pool_w = pool_w # プーリングの幅
        self.stride = stride # プーリングのストライド
        self.pad = pad # プーリングのパディング

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = (H - self.pool_h) // self.stride + 1
        out_w = (W - self.pool_w) // self.stride + 1

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w

        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(dout.size), np.argmax(self.col, axis=1)] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(-1, pool_size)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
    

if __name__ == "__main__":
    # 簡単なテストケース
    np.random.seed(0)
    x = np.random.rand(1, 1, 4, 4) # 入力データ
    W = np.random.rand(1, 1, 3, 3) # フィルタの重み
    b = np.random.rand(1) # フィルタのバイアス

    conv = Convolution(W, b)
    out = conv.forward(x)
    print("Convolution Forward Output:\n", out)

    # dout = np.random.rand(*out.shape) # 上流からの勾配
    # dx, dW, db = conv.backward(dout)
    # print("Convolution Backward dx:\n", dx)
    # print("Convolution Backward dW:\n", dW)
    # print("Convolution Backward db:\n", db)

    pool = MaxPooling(pool_h=2, pool_w=2, stride=2)
    out_pool = pool.forward(out)
    print("MaxPooling Forward Output:\n", out_pool)

    # dout_pool = np.random.rand(*out_pool.shape) # 上流からの勾配
    # dx_pool = pool.backward(dout_pool)
    # print("MaxPooling Backward dx:\n", dx_pool)