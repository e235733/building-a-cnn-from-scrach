import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from process import Convolution, MaxPooling

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_all = mnist.data.reshape(-1, 1, 28, 28) / 255.0  # 正規化

sample_img = X_all[4:5] # (1, 1, 28, 28)


filter_v = np.array([[[
    [-1, 2, -1],
    [-1, 2, -1],
    [-1, 2, -1],
]]])

filter_h = np.array([[[
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1],
]]])

filter_some = np.array([[[
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
]]])

conv_v = Convolution(filter_v, 0)
conv_h = Convolution(filter_h, 0)
conv_some = Convolution(filter_some, 0)
out_v = conv_v.forward(sample_img)
out_h = conv_h.forward(sample_img)
out_some = conv_some.forward(sample_img)

pool = MaxPooling(2, 2)
pool_raw = pool.forward(sample_img)
pool_v = pool.forward(out_v)
pool_h = pool.forward(out_h)
pool_some = pool.forward(out_some)

fig, axes = plt.subplots(2, 4, figsize=(16, 4))

# 元画像
axes[0, 0].imshow(sample_img[0, 0], cmap='gray')
axes[0, 0].set_title("Original MNIST")
axes[0, 0].axis('off')

# 縦エッジ
axes[0, 1].imshow(out_v[0, 0], cmap='gray')
axes[0, 1].set_title("Vertical Edge Detection")
axes[0, 1].axis('off')

# 横エッジ
axes[0, 2].imshow(out_h[0, 0], cmap='gray')
axes[0, 2].set_title("Horizontal Edge Detection")
axes[0, 2].axis('off')

axes[0, 3].imshow(out_some[0, 0], cmap='gray')
axes[0, 3].set_title("Something")
axes[0, 3].axis('off')

axes[1, 0].imshow(pool_raw[0, 0], cmap='gray')
axes[1, 0].set_title("")
axes[1, 0].axis('off')

axes[1, 1].imshow(pool_v[0, 0], cmap='gray')
axes[1, 1].set_title("")
axes[1, 1].axis('off')

axes[1, 2].imshow(pool_h[0, 0], cmap='gray')
axes[1, 2].set_title("")
axes[1, 2].axis('off')

axes[1, 3].imshow(pool_some[0, 0], cmap='gray')
axes[1, 3].set_title("")
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()