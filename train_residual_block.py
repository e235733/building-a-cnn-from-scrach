"""ResidualBlock を使った学習テスト"""

from mnist_dataset import MnistDataset
from nn import NN_Model
from common.trainer import Trainer
from plotter import Plotter

def main():
    print("\n" + "=" * 60)
    print("ResidualBlock を使ったMNIST学習テスト")
    print("=" * 60 + "\n")
    
    # --- ハイパーパラメータの設定 ---
    n_samples = 10000  # テスト用に小さめ
    epochs = 3
    mini_batch_size = 128
    optimizer = 'Momentum'
    optimizer_param = {'lr': 0.01}
    evaluate_sample_num_per_epoch = 2000
    
    # --- データの準備 ---
    print("Loading MNIST dataset...")
    dataset = MnistDataset(n_samples=n_samples)
    
    # CNN用にデータを(N, C, H, W)に整形
    x_train = dataset.X_train.reshape(-1, 1, 28, 28)
    x_test = dataset.X_test.reshape(-1, 1, 28, 28)
    t_train = dataset.Y_train
    t_test = dataset.Y_test
    
    print(f"  Train data: {x_train.shape}")
    print(f"  Test data:  {x_test.shape}\n")

    # --- モデルの構築（ResidualBlock付き） ---
    print("Building model with ResidualBlock...")
    layer_config = [
        ['Conv', 16, 5, 2, 1],              # Conv(1->16)
        ['Relu'],
        ['ResidualBlock', 16, 3, 1, 1],     # ResidualBlock(同チャネル)
        ['Pool', 2, 2, 2],                  # Pooling
        ['ResidualBlock', 32, 3, 2, 1],     # ResidualBlock(チャネル増加+stride=2)
        ['Pool', 2, 2, 2],                  # Pooling
        ['Affine', 100],                    # 全結合層
        ['Relu'],
        ['Affine', 10]                      # 出力層
    ]
    
    model = NN_Model(
        input_dim=(1, 28, 28),
        layer_config=layer_config,
        weight_init_type='he',
        use_batchnorm=False
    )
    
    print(f"Model layers: {len(model.layers)}")
    print(f"Model parameters: {len(model.params)}\n")

    # --- トレーナーの準備 ---
    print("Starting training...\n")
    trainer = Trainer(
        model, x_train, t_train, x_test, t_test,
        epochs=epochs, 
        mini_batch_size=mini_batch_size,
        optimizer=optimizer, 
        optimizer_param=optimizer_param,
        evaluate_sample_num_per_epoch=evaluate_sample_num_per_epoch,
        verbose=True
    )

    # --- 訓練 ---
    trainer.train()

    # --- 最終結果 ---
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Final train accuracy: {trainer.train_acc_list[-1]:.4f}")
    print(f"Final test accuracy:  {trainer.test_acc_list[-1]:.4f}")
    print("\n")


if __name__ == "__main__":
    main()
