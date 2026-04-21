import numpy as np
import matplotlib.pyplot as plt

class ToyCrossDataset:
    """
    CNNの動作検証用：2x2の極小トイデータセットを生成するクラス
    正解(1): (0,1)と(1,0)が相対的に黒く、(0,0)と(1,1)が白いパターン（斜め線/クロス）
    不正解(0): それ以外のパターン
    """
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def _is_correct_pattern(self, img):
        """画像が正解の条件を満たしているか判定する（内部用メソッド）"""
        black_positions = [img[0, 1], img[1, 0]]
        white_positions = [img[0, 0], img[1, 1]]
        return min(black_positions) > max(white_positions)

    def load_data(self):
        """
        データセットを生成して返す
        Returns:
            X: (N, 1, 2, 2) の入力データ（0.0~1.0に正規化済み）
            Y: (N,) の正解ラベル（0 または 1）
        """
        X = np.zeros((self.num_samples, 1, 2, 2), dtype=np.float32)
        Y = np.zeros(self.num_samples, dtype=int)
        
        for i in range(self.num_samples):
            img = np.zeros((2, 2), dtype=np.float32)
            base_black = np.random.randint(180, 240)
            base_white = np.random.randint(20, 80)
            
            # 半分を正解データ、半分を不正解データにする
            if i % 2 == 0:
                # 正解データの生成
                img[0, 1] = np.clip(base_black + np.random.randint(-20, 21), 0, 255)
                img[1, 0] = np.clip(base_black + np.random.randint(-20, 21), 0, 255)
                img[0, 0] = np.clip(base_white + np.random.randint(-20, 21), 0, 255)
                img[1, 1] = np.clip(base_white + np.random.randint(-20, 21), 0, 255)
                Y[i] = 1
                
            else:
                # 不正解データの生成（条件を満たさなくなるまで再生成）
                while True:
                    for r in range(2):
                        for c in range(2):
                            tone = np.random.choice(['black', 'white', 'gray'])
                            if tone == 'black':
                                val = base_black + np.random.randint(-10, 11)
                            elif tone == 'white':
                                val = base_white + np.random.randint(-10, 11)
                            else:
                                val = np.random.randint(100, 160)
                            img[r, c] = np.clip(val, 0, 255)
                    
                    if not self._is_correct_pattern(img):
                        break
                Y[i] = 0
                
            X[i, 0, :, :] = img

        # 0.0〜1.0に正規化
        return X / 255.0, Y

    def plot_samples(self, X, Y, num_show=2):
        """
        生成したデータを視覚化するユーティリティメソッド
        """
        # 正解と不正解をそれぞれ取得
        correct_idx = np.where(Y == 1)[0][:num_show]
        incorrect_idx = np.where(Y == 0)[0][:num_show]
        
        fig, axes = plt.subplots(2, num_show, figsize=(3 * num_show, 6))
        
        for i in range(num_show):
            # 正解の描画（上段）
            if i < len(correct_idx):
                axes[0, i].imshow(X[correct_idx[i], 0], cmap='gray', vmin=0, vmax=1)
                axes[0, i].set_title("Correct (Y=1)")
                axes[0, i].axis('off')
                
            # 不正解の描画（下段）
            if i < len(incorrect_idx):
                axes[1, i].imshow(X[incorrect_idx[i], 0], cmap='gray', vmin=0, vmax=1)
                axes[1, i].set_title("Incorrect (Y=0)")
                axes[1, i].axis('off')
                
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    generator = ToyCrossDataset(100)
    X, Y = generator.load_data()
    generator.plot_samples(X, Y, 8)