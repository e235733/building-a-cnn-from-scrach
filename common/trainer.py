from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()
from common.optimizer import *

class Trainer:
    def __init__(self, model, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.model = model
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        
        # 学習開始前（真っ新な状態）の精度を記録
        self._evaluate(display_epoch=0)

    def _evaluate(self, display_epoch):
        # 評価モード設定
        if hasattr(self.model, 'is_training'):
            self.model.is_training = False
            
        x_train_sample, t_train_sample = self.x_train, self.t_train
        x_test_sample, t_test_sample = self.x_test, self.t_test
        if not self.evaluate_sample_num_per_epoch is None:
            t = self.evaluate_sample_num_per_epoch
            x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
            x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
            
        train_acc = self.model.accuracy(x_train_sample, t_train_sample)
        test_acc = self.model.accuracy(x_test_sample, t_test_sample)

        if hasattr(train_acc, 'get'):
            train_acc = train_acc.get()
        if hasattr(test_acc, 'get'):
            test_acc = test_acc.get()

        self.train_acc_list.append(train_acc)
        self.test_acc_list.append(test_acc)

        if self.verbose: 
            print(f"=== epoch: {display_epoch}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f} ===")

    def train_step(self):
        batch_mask = xp.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        x_batch = xp.asarray(x_batch)
        t_batch = xp.asarray(t_batch)

        # 訓練モード設定
        if hasattr(self.model, 'is_training'):
            self.model.is_training = True

        grads = self.model.gradient(x_batch, t_batch)
        self.optimizer.update(self.model.params, grads)
        
        loss = self.model.last_loss
        self.train_loss_list.append(loss)
        
        self.current_iter += 1

        # 各エポックの終了時に評価
        if self.current_iter % self.iter_per_epoch == 0 or self.current_iter == self.max_iter:
            display_epoch = self.current_iter // self.iter_per_epoch
            # 最終ステップが中途半端な場合でも最大エポック数を超えないように
            display_epoch = min(display_epoch, self.epochs)
            self._evaluate(display_epoch)

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.model.accuracy(self.x_test, self.t_test)
        if hasattr(test_acc, 'get'):
            test_acc = test_acc.get()

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
