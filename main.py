import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mnist_dataset import MnistDataset
from cnn import CNN_Model
from common.trainer import Trainer
from plotter import Plotter

def main():
    # --- ハイパーパラメータの設定 ---
    # 学習を早く終わらせるため、デフォルトではサンプル数を絞っています
    # 実力を見たい場合は n_samples=70000 にしてください
    n_samples = 5000 
    epochs = 20
    mini_batch_size = 100
    optimizer = 'Momentum'
    optimizer_param = {'lr': 0.01}
    evaluate_sample_num_per_epoch = 1000
    
    # --- データの準備 ---
    dataset = MnistDataset(n_samples=n_samples)
    
    # CNN用にデータを(N, C, H, W)に整形
    x_train = dataset.X_train.reshape(-1, 1, 28, 28)
    x_test = dataset.X_test.reshape(-1, 1, 28, 28)
    t_train = dataset.Y_train
    t_test = dataset.Y_test

    # --- モデルの構築 ---
    print("Initializing CNN...")
    model = CNN_Model(
        input_dim=(1, 28, 28),
        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=100, 
        output_size=10, 
        weight_init_std=0.01
    )

    # --- トレーナーの準備 ---
    trainer = Trainer(
        model, x_train, t_train, x_test, t_test,
        epochs=epochs, 
        mini_batch_size=mini_batch_size,
        optimizer=optimizer, 
        optimizer_param=optimizer_param,
        evaluate_sample_num_per_epoch=evaluate_sample_num_per_epoch,
        verbose=True
    )

    # --- プロッターの準備 ---
    # X は平坦化された状態で渡す（Plotter内部で次元を判断するため）
    plotter = Plotter(interval=0.1, X=dataset.X_train[:500], Y=t_train[:500], is_detail_mode=True)

    # --- 学習の実行 ---
    print("Start Training...")
    # 学習の各ステップで描画を更新したい場合は、trainer.train() を使わずにループを回します
    for i in range(trainer.max_iter):
        trainer.train_step()
        
        # エポックごとにプロットを更新
        if trainer.current_iter % trainer.iter_per_epoch == 0:
            plotter.show(trainer)

    print("Training Finished.")

    # --- 結果の可視化と評価 ---
    # 最終的な正解率の表示
    plotter.show(trainer)
    
    # 評価用画面（混同行列など）
    print("Showing Evaluation...")
    plotter.show_evaluation(model, x_test[:1000], t_test[:1000])
    
    # フィルターの可視化
    print("Visualizing CNN Filters...")
    plotter.visualize_filters(model)
    
    plotter.finish()

if __name__ == '__main__':
    main()
