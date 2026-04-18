from mnist_dataset import MnistDataset
from plotter import Plotter
from data_loader import DataLoader, DataNormalizer
from nn import NN_Model
import function as fn
import numpy as np
from sklearn.model_selection import train_test_split

import time

def main():

    NUM_DATA = 70000
    NUM_EPOCHS = 100
    BATCH_SIZE = 512

    # モデル構成
    HIDDEN_LAYER = [256, 128, 64, 16]
    
    # 活性化関数
    ACT_FUNCTION = fn.LeakyReLU()
    OUTPUT_FUNCTION = fn.Softmax()

    # ハイパーパラメータ
    ETA = 0.01  # 学習率
    L2_LAMBDA = 0.005  # L2正則化のペナルティ
    ALPHA = 0.9  # Momentum の慣性係数

    # チェック時やデバッグ時はTrue
    IS_DETAIL_MODE = True

    # --- データの準備 ---
    all_data = MnistDataset(NUM_DATA)

    X_train = all_data.X_train
    Y_train = all_data.Y_train
    X_test = all_data.X_test
    Y_test = all_data.Y_test

    # テストデータのサイズ
    test_data_size = 10000

    # # データをランダムに訓練用とテスト用に分割
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     all_data.X, all_data.Y, test_size=test_data_size, random_state=42
    # )

    normalizer = DataNormalizer(X_train)
    X_train_norm = normalizer.normalize(X_train)
    X_test_norm = normalizer.normalize(X_test)

    train_loader = DataLoader(X_train_norm, Y_train, batch_size=BATCH_SIZE)

    # モデルのインスタンス化
    input_dim = X_train_norm.shape[1]
    output_dim = Y_train.shape[1] if Y_train.ndim > 1 else int(np.max(Y_train) + 1)

    model = NN_Model(
        input_dim=input_dim,
        hidden_layer=HIDDEN_LAYER,
        output_dim=output_dim,
        act_fn=ACT_FUNCTION,
        output_fn=OUTPUT_FUNCTION,
        eta=ETA,
        l2_lambda=L2_LAMBDA,
        alpha=ALPHA
    )
    
    # プロッターの初期化（正規化済みのデータを渡す）
    plotter = Plotter(0.1, X_train_norm[:500], Y_train[:500], IS_DETAIL_MODE)

    print(f"Start training: {len(X_train)} samples, {len(train_loader)} batches per epoch")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):

        for X_batch, Y_batch in train_loader:
            model.shift(X_batch, Y_batch)
        
        # 損失の記録と表示
        if epoch % 10 == 0:
            train_loss = model.log_train_loss(X_train_norm[:test_data_size], Y_train[:test_data_size])
            test_loss = model.log_test_loss(X_test_norm, Y_test)
            
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            plotter.show(model)

    end_time = time.time()
            
    plotter.show(model)
    plotter.show_evaluation(model, X_test_norm, Y_test)

    # 最終的な正解率の計算
    train_accuracy = model.evaluate_accuracy(X_train_norm, Y_train)
    print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%")

    test_accuracy = model.evaluate_accuracy(X_test_norm, Y_test)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    plotter.finish()

    print("time:", end_time - start_time)

if __name__ == "__main__":
    main()
