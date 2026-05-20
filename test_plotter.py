from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()
import time
from toy_closs_dataset import ToyClossDataset
from nn import NN_Model
from plotter import Plotter
from common.trainer import Trainer

def test_plotter_live():
    # --- データ生成 (2次元) ---
    dataset = ToyClossDataset(num_classes=3, points_per_class=100)
    X = dataset.X
    Y = dataset.Y
    
    # --- モデル構築 ---
    model = NN_Model(input_dim=(2,), layer_config=[['Affine', 10], ['Relu'], ['Affine', 3]])
    
    # --- トレーナー設定 ---
    trainer = Trainer(model, X, Y, X, Y, epochs=20, mini_batch_size=30, optimizer='Momentum', optimizer_param={'lr': 0.1}, verbose=False)
    
    # --- プロッター設定 ---
    plotter = Plotter(interval=0.01, X=X, Y=Y, is_detail_mode=True)
    
    # --- 学習ループ ---
    print("Starting live plot test...")
    for i in range(trainer.max_iter):
        trainer.train_step()
        
        # 定期的またはエポックごとに表示を更新
        if i % 10 == 0:
            plotter.show(trainer)
            
    print("Test finished.")
    plotter.finish()

if __name__ == "__main__":
    test_plotter_live()
