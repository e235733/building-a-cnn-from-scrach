import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataNormalizer

class Plotter:
    def __init__(self, interval, X, Y, is_detail_mode=False):
        self.interval = interval
        self.X = X
        # one-hot to integer labels
        self.Y_labels = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        self.num_classes = int(np.max(self.Y_labels) + 1)
        self.is_detail_mode = is_detail_mode
        self.input_dim = X.shape[1]
        
        if self.is_detail_mode:
            # Detail mode: 2x3 grid.
            self.fig, self.axs = plt.subplots(2, 3, figsize=(12, 7))
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
            self.ax_loss = self.axs[0, 0]
            self.ax_data = self.axs[0, 1]
            if self.input_dim not in [1, 2]:
                self.ax_data.set_axis_off()
                self.ax_data.set_title(f"No plot ({self.input_dim}D)")
        else:
            # Default mode: 1x2 if 1D or 2D, 1x1 if not
            if self.input_dim in [1, 2]:
                self.fig = plt.figure(figsize=(12, 6))
                self.ax_loss = self.fig.add_subplot(1, 2, 1)
                self.ax_data = self.fig.add_subplot(1, 2, 2)
            else:
                self.fig = plt.figure(figsize=(6, 6))
                self.ax_loss = self.fig.add_subplot(1, 1, 1)
                self.ax_data = None

    def show(self, trainer):
        self._show_loss(trainer)
        self._show_accuracy(trainer)
        
        # モデルの予測境界を表示（1D/2Dデータの場合）
        if self.input_dim == 2:
            self.ax_data.cla()
            self._plot_2d(trainer.model, self.ax_data)
        elif self.input_dim == 1:
            self.ax_data.cla()
            self._plot_1d(trainer.model, self.ax_data)
            
        if self.is_detail_mode:
            self._show_network_stats(trainer.model)
        
        plt.pause(self.interval)

    def _show_loss(self, trainer):
        self.ax_loss.cla()
        self.ax_loss.plot(trainer.train_loss_list, color='purple', linewidth=2, label='Train Loss')
        self.ax_loss.set_title("Learning Curve (Loss)")
        self.ax_loss.set_xlabel("Iteration")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)

    def _show_accuracy(self, trainer):
        # 詳細モードの場合、別の軸にAccuracyを表示する
        # 現状の2x3レイアウトを活かすため、特定のaxをAccuracy用に割り当てるか、
        # 軸を共有（twinx）するなどの工夫が必要。
        # ここでは簡単のため、もしAccuracyデータがあればLossと同じグラフに重ねるか
        # 別のウィンドウ（または空いているax）を使います。
        if not hasattr(trainer, 'train_acc_list'):
            return
            
        # 仮に詳細モードの ax_data が使われていない（高次元データ）場合、そこを Accuracy グラフにする
        if self.input_dim not in [1, 2] and self.ax_data is not None:
            self.ax_data.cla()
            self.ax_data.plot(trainer.train_acc_list, label='train acc')
            self.ax_data.plot(trainer.test_acc_list, label='test acc', linestyle='--')
            self.ax_data.set_title("Accuracy")
            self.ax_data.set_xlabel("Epochs")
            self.ax_data.set_ylabel("Accuracy")
            self.ax_data.legend(loc='lower right')
            self.ax_data.grid(True)

    def _show_network_stats(self, model):
        ax_w = self.axs[1, 0]
        ax_A = self.axs[1, 1]
        ax_dw = self.axs[1, 2]
        
        ax_w.cla()
        ax_A.cla()
        ax_dw.cla()

        grad_names = []
        grad_means = []
        has_w_plot = False
        has_A_plot = False
        
        for name, layer in model.layers.items():
            # 重みの分布 (Affine, Convolutionなど)
            if hasattr(layer, 'W') and layer.W is not None:
                ax_w.hist(layer.W.flatten(), bins=30, alpha=0.5, label=name)
                has_w_plot = True
            
            # 活性化値の分布 (各レイヤーの出力)
            # 畳み込み層などの多次元出力はflattenして表示
            if hasattr(layer, 'out') and layer.out is not None:
                ax_A.hist(layer.out.flatten(), bins=30, alpha=0.5, label=name)
                has_A_plot = True
            
            # 勾配の大きさ
            if hasattr(layer, 'dW') and layer.dW is not None:
                grad_names.append(name)
                grad_means.append(np.mean(np.abs(layer.dW)))
        
        ax_w.set_title("Weight Distribution")
        if has_w_plot:
            ax_w.legend(fontsize='x-small')
        
        ax_A.set_title("Activation Distribution")
        if has_A_plot:
            ax_A.legend(fontsize='x-small')
        
        ax_dw.set_title("Mean Gradient Magnitude (|dW|)")
        if grad_means:
            x_pos = np.arange(len(grad_names))
            ax_dw.bar(x_pos, grad_means, color='orange')
            ax_dw.set_xticks(x_pos)
            ax_dw.set_xticklabels(grad_names, rotation=45, ha='right', fontsize='small')

    # 混同行列と誤認識データの詳細を表示する
    def show_evaluation(self, model, X_test, Y_test):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.gridspec as gridspec

        probs = model.predict(X_test)
        Y_pred = np.argmax(probs, axis=1)
        Y_true = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test

        # 誤認識したデータのインデックスを取得
        error_indices = np.where(Y_pred != Y_true)[0]

        fig_eval = plt.figure(num="Evaluation Results", figsize=(14, 6))
        gs = gridspec.GridSpec(3, 6, figure=fig_eval)

        # 左側：混同行列
        ax_cm = fig_eval.add_subplot(gs[:, :3])
        cm = confusion_matrix(Y_true, Y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title("Confusion Matrix", fontsize=14)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")

        # 右側：誤認識データ
        if len(error_indices) == 0:
            ax_msg = fig_eval.add_subplot(gs[:, 3:])
            ax_msg.text(0.5, 0.5, "Perfect! 0 Errors.", ha='center', va='center', fontsize=20)
            ax_msg.axis('off')
            return

        # MNIST の場合 (784次元)
        if self.input_dim == 784:
            img_size = int(np.sqrt(self.input_dim))
            num_show = min(9, len(error_indices))
            
            for i in range(num_show):
                idx = error_indices[i]
                row = i // 3
                col = 3 + (i % 3)
                
                ax_img = fig_eval.add_subplot(gs[row, col])
                ax_img.imshow(X_test[idx].reshape(img_size, img_size), cmap='gray')
                ax_img.set_title(f"T:{Y_true[idx]} $\\rightarrow$ P:{Y_pred[idx]}", color='red', fontsize=10)
                ax_img.axis('off')
                
        # 2次元データの場合
        elif self.input_dim == 2:
            ax_err = fig_eval.add_subplot(gs[:, 3:])
            ax_err.scatter(X_test[:, 0], X_test[:, 1], c=Y_true, cmap='tab10', alpha=0.2, label='All Test Data')
            ax_err.scatter(X_test[error_indices, 0], X_test[error_indices, 1], 
                           color='red', marker='x', s=100, linewidth=2, label='Misclassified')
            ax_err.set_title("Error Distribution", fontsize=14)
            ax_err.legend()
            
        else:
            # 1次元やその他の次元の場合
            ax_txt = fig_eval.add_subplot(gs[:, 3:])
            ax_txt.axis('off')
            msg = "Misclassified Samples (Top 10):\n\n"
            num_show = min(10, len(error_indices))
            for i in range(num_show):
                idx = error_indices[i]
                msg += f"Index {idx}: True={Y_true[idx]}, Pred={Y_pred[idx]}\n"
            ax_txt.text(0.1, 0.9, msg, va='top', fontfamily='monospace')

    def finish(self):
        plt.show()

    def _plot_1d(self, model, ax):
        ax.set_title(f"Decision Boundary (1D, {self.num_classes} classes)")
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        
        probs = model.predict(xx)
        
        cmap = plt.get_cmap('tab10')
        for c in range(self.num_classes):
            ax.plot(xx, probs[:, c], color=cmap(c), label=f'Prob Class {c}')
            
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        
        ax.scatter(self.X[:, 0], np.zeros_like(self.Y_labels), c=self.Y_labels, cmap='tab10', edgecolors='k', zorder=3, label='True Labels')
        
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Input X")
        ax.set_ylabel("Probability")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)

    def _plot_2d(self, model, ax):
        ax.set_title(f"Decision Boundary ({self.num_classes} classes)")
        
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        probs = model.predict(grid_points)
        predicted_classes = np.argmax(probs, axis=1)
        predicted_grid = predicted_classes.reshape(xx.shape)

        ax.contourf(xx, yy, predicted_grid, alpha=0.3, cmap='tab10', levels=np.arange(self.num_classes + 1) - 0.5)
        
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.Y_labels, cmap='tab10', edgecolors='k')
