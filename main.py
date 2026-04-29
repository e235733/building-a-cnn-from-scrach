from mnist_dataset import MnistDataset
from cnn import CNN_Model
from common.trainer import Trainer
from plotter import Plotter

def main():
    # --- ハイパーパラメータの設定 ---
    n_samples = 70000
    epochs = 20
    mini_batch_size = 1000
    optimizer = 'Momentum'
    optimizer_param = {'lr': 0.05}
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
        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 2, 'stride': 1},
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
    plotter = Plotter(interval=0.1, X=dataset.X_train[:500], Y=t_train[:500], is_detail_mode=True)

    # --- 学習の実行 ---
    print("Start Training...")
    for _ in range(trainer.max_iter):
        trainer.train_step()
        
        # エポックごとにプロットを更新
        if trainer.current_iter % trainer.iter_per_epoch == 0:
            plotter.show(trainer)

    print("Training Finished.")

    # --- 結果の可視化と評価 ---
    plotter.show(trainer)
    
    print("Showing Evaluation...")
    plotter.show_evaluation(model, x_test[:1000], t_test[:1000])
    
    print("Visualizing CNN Filters...")
    plotter.visualize_filters(model)
    
    plotter.finish()

if __name__ == '__main__':
    main()
