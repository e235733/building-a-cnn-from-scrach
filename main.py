from mnist_dataset import MnistDataset
from cifar10_dataset import Cifar10Dataset
from nn import NN_Model
from common.layer_spec import *
from common.trainer import Trainer
from plotter import Plotter

def main():
    # --- ハイパーパラメータの設定 ---
    n_samples = 200  # Noneの場合は全データを使用
    epochs = 10
    mini_batch_size = 128
    optimizer = 'Momentum'
    optimizer_param = {'lr': 0.1, 'weight_decay': 1e-5}
    evaluate_sample_num_per_epoch = 10000
    
    # --- データの準備 ---
    dataset = MnistDataset(n_samples=n_samples)
    
    # CNN用にデータを(N, C, H, W)に整形
    x_train = dataset.X_train.reshape(-1, 1, 28, 28)
    x_test = dataset.X_test.reshape(-1, 1, 28, 28)
    t_train = dataset.Y_train
    t_test = dataset.Y_test

    # --- モデルの構築 ---
    print("Initializing CNN...")
    model = NN_Model(layer_config=[ResidualBlock(16, 3),
                                   ResidualBlock(16, 3),
                                   Pool(),
                                   ResidualBlock(32, 3),
                                   ResidualBlock(32, 3),
                                   Pool(),
                                   ResidualBlock(64, 3),
                                   ResidualBlock(64, 3),
                                   BatchNorm(), Relu(),
                                   Conv(64, 3, 1, 0),
                                   BatchNorm(), Relu(),
                                   Conv(64, 3, 1, 0),
                                   BatchNorm(), Relu(),
                                   GAP(),
                                   Affine(10)])

    # --- トレーナーの準備 ---
    trainer = Trainer(
        model, x_train, t_train, x_test, t_test,
        epochs=epochs, 
        mini_batch_size=mini_batch_size,
        optimizer=optimizer, 
        optimizer_param=optimizer_param,
        evaluate_sample_num_per_epoch=evaluate_sample_num_per_epoch,
        verbose=True,
        log_interval=50
    )

    # --- プロッターの準備 ---
    plotter = Plotter(interval=0.1, X=dataset.X_train[:500], Y=t_train[:500], is_detail_mode=True)

    # --- 学習の実行 ---
    print("Start Training...")
    for _ in range(trainer.max_iter):
        trainer.train_step()
        
    print("Training Finished.")

    # --- 結果の可視化と評価 ---
    plotter.show(trainer, save_path="training_result.png")
    
    print("Showing Evaluation...")
    plotter.show_evaluation(model, x_test[:1000], t_test[:1000], save_path="evaluation.png")
    
    print("Visualizing CNN Filters...")
    plotter.visualize_filters(model, save_path="filters.png")
    
    plotter.finish(save_path="final_plot.png")

if __name__ == '__main__':
    main()
