from cifar10_dataset import Cifar10Dataset
from nn import NN_Model
from common.trainer import Trainer
from plotter import Plotter
from common import config

def main():
    # --- ハイパーパラメータの設定 ---
    n_samples = None
    epochs = 20
    mini_batch_size = 128
    optimizer = 'Momentum' 
    optimizer_param = {'lr': 0.05, 'weight_decay': 1e-5}
    evaluate_sample_num_per_epoch = 10000
    
    # --- データの準備 ---
    dataset = Cifar10Dataset(n_samples=n_samples)
    
    # CNN用にデータを(N, C, H, W)に整形 (CIFAR-10は3チャンネル)
    x_train = dataset.X_train.reshape(-1, 3, 32, 32)
    x_test = dataset.X_test.reshape(-1, 3, 32, 32)
    t_train = dataset.Y_train
    t_test = dataset.Y_test

    # --- モデルの構築 ---
    print("Initializing CNN for CIFAR-10...")
    model = NN_Model(input_dim=(3, 32, 32),
                     layer_config=[['ResidualBlock', 32, 3, 1, 1],
                                   ['ResidualBlock', 32, 3, 1, 1],
                                   ['Pool', 2, 2, 2],
                                   ['ResidualBlock', 64, 3, 1, 1],
                                   ['ResidualBlock', 64, 3, 1, 1],
                                   ['Pool', 2, 2, 2],
                                   ['ResidualBlock', 128, 3, 1, 1],
                                   ['ResidualBlock', 128, 3, 1, 1],
                                   ['BatchNorm'], ['Relu'],
                                   ['Conv', 128, 3, 0, 1],
                                   ['BatchNorm'], ['Relu'],
                                   ['Conv', 128, 3, 0, 1],
                                   ['BatchNorm'], ['Relu'],
                                   ['Affine', 10]])

    # --- トレーナーの準備 ---
    trainer = Trainer(
        model, x_train, t_train, x_test, t_test,
        epochs=epochs, 
        mini_batch_size=mini_batch_size,
        optimizer=optimizer, 
        optimizer_param=optimizer_param,
        evaluate_sample_num_per_epoch=evaluate_sample_num_per_epoch,
        verbose=True,
        log_interval=20
    )

    # --- プロッターの準備 ---
    plotter = Plotter(interval=0.1, X=dataset.X_train[:500], Y=t_train[:500], is_detail_mode=True)

    # --- 学習の実行 ---
    print(f"Start Training (GPU: {config.GPU_ENABLE})...")
    for _ in range(trainer.max_iter):
        trainer.train_step()
        
    print("Training Finished.")

    # --- 結果の可視化と評価 ---
    plotter.show(trainer, save_path="cifar10_result.png")
    
    print("Showing Evaluation...")
    plotter.show_evaluation(model, x_test[:1000], t_test[:1000], save_path="cifar10_evaluation.png")
    
    # フィルターの可視化
    print("Visualizing CNN Filters...")
    plotter.visualize_composed_filters(model, x_test[:32], top_k=16, save_path="cifar10_composed_filters.png")

    # クラスごとの復元画像（GAP -> Affine構成が前提。現在のlayer_configはGAP未使用のため未対応）
    print("Visualizing Class Reconstructions...")
    plotter.visualize_class_reconstructions(model, class_names=dataset.label_names, save_path="cifar10_class_reconstructions.png")

    plotter.finish(save_path="cifar10_final_plot.png")

if __name__ == '__main__':
    main()
