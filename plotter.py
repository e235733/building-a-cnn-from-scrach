import matplotlib.pyplot as plt
import numpy as np

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
            self.fig, self.axs = plt.subplots(2, 3, figsize=(15, 9))
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
            self.ax_loss = self.axs[0, 0]
            self.ax_data = self.axs[0, 1]
            self.ax_acc = self.axs[0, 2]
            
            if self.input_dim != 2:
                self.ax_data.set_axis_off()
                self.ax_data.set_title(f"No 2D Plot ({self.input_dim}D)")
        else:
            # Default mode
            if self.input_dim == 2:
                self.fig = plt.figure(figsize=(12, 6))
                self.ax_loss = self.fig.add_subplot(1, 2, 1)
                self.ax_data = self.fig.add_subplot(1, 2, 2)
                self.ax_acc = None
            else:
                self.fig = plt.figure(figsize=(12, 6))
                self.ax_loss = self.fig.add_subplot(1, 2, 1)
                self.ax_acc = self.fig.add_subplot(1, 2, 2)
                self.ax_data = None

    def show(self, trainer):
        # trainerかmodelかを自動判別
        model = getattr(trainer, 'model', trainer)
        
        self._show_loss(trainer)
        self._show_accuracy(trainer)
        
        # モデルの予測境界を表示（2Dデータの場合のみ）
        if self.input_dim == 2 and self.ax_data is not None:
            self.ax_data.cla()
            self._plot_2d(model, self.ax_data)
        
        # 詳細統計情報の描画
        if self.is_detail_mode:
            self._show_network_stats(model)
        
        plt.pause(self.interval)

    def _show_loss(self, trainer):
        self.ax_loss.cla()
        # いろいろな属性名に対応
        loss_list = getattr(trainer, 'train_loss_list', 
                    getattr(trainer, 'loss_history', 
                    getattr(trainer, 'train_loss_history', [])))
        
        if len(loss_list) > 0:
            self.ax_loss.plot(loss_list, color='purple', linewidth=1, label='Train Loss')
            self.ax_loss.set_title("Learning Curve (Loss)")
            self.ax_loss.set_xlabel("Iteration")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.legend()
            self.ax_loss.grid(True)

    def _show_accuracy(self, trainer):
        if self.ax_acc is None:
            return
            
        self.ax_acc.cla()
        train_acc = getattr(trainer, 'train_acc_list', [])
        test_acc = getattr(trainer, 'test_acc_list', [])
        
        if len(train_acc) > 0:
            epochs = np.arange(len(train_acc))
            self.ax_acc.plot(epochs, train_acc, label='train acc', marker='o', markersize=3)
            if len(test_acc) > 0 and len(test_acc) == len(train_acc):
                self.ax_acc.plot(epochs, test_acc, label='test acc', linestyle='--', marker='x', markersize=3)
            self.ax_acc.set_title("Accuracy Transition")
            self.ax_acc.set_xlabel("Epochs")
            self.ax_acc.set_ylabel("Accuracy")
            self.ax_acc.set_ylim(0, 1.05)
            self.ax_acc.legend(loc='lower right')
            self.ax_acc.grid(True)
        else:
            self.ax_acc.set_title("Accuracy (No Data)")
            if self.is_detail_mode:
                self.ax_acc.set_axis_off()

    def _show_network_stats(self, model):
        if not hasattr(model, 'layers'):
            return
            
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
            if hasattr(layer, 'W') and layer.W is not None:
                ax_w.hist(layer.W.flatten(), bins=30, alpha=0.5, label=name)
                has_w_plot = True
            
            if hasattr(layer, 'out') and layer.out is not None:
                ax_A.hist(layer.out.flatten(), bins=30, alpha=0.5, label=name)
                has_A_plot = True
            
            if hasattr(layer, 'dW') and layer.dW is not None:
                grad_names.append(name)
                grad_means.append(np.mean(np.abs(layer.dW)))
        
        ax_w.set_title("Weight Distribution")
        if has_w_plot: ax_w.legend(fontsize='x-small')
        
        ax_A.set_title("Activation Distribution")
        if has_A_plot: ax_A.legend(fontsize='x-small')
        
        ax_dw.set_title("Mean Gradient Magnitude (|dW|)")
        if grad_means:
            x_pos = np.arange(len(grad_names))
            ax_dw.bar(x_pos, grad_means, color='orange')
            ax_dw.set_xticks(x_pos)
            ax_dw.set_xticklabels(grad_names, rotation=45, ha='right', fontsize='small')

    def _plot_2d(self, model, ax):
        ax.set_title(f"Decision Boundary (2D)")
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        probs = model.predict(grid_points)
        predicted_classes = np.argmax(probs, axis=1)
        predicted_grid = predicted_classes.reshape(xx.shape)
        ax.contourf(xx, yy, predicted_grid, alpha=0.3, cmap='tab10')
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.Y_labels, cmap='tab10', edgecolors='k')

    def visualize_filters(self, model, title="CNN Filters"):
        # 最初のConvolution層を探す
        conv_layer = None
        for layer in model.layers.values():
            from common.layers import Convolution
            if isinstance(layer, Convolution):
                conv_layer = layer
                break
        
        if conv_layer is None or not hasattr(conv_layer, 'W'):
            print("No Convolutional layer with weights found in the model.")
            return

        FN, C, FH, FW = conv_layer.W.shape
        ny = int(np.ceil(np.sqrt(FN)))
        nx = int(np.ceil(FN / ny))
        
        # フィルター間に1ピクセルの隙間を空けるための計算
        margin = 1
        grid_h = ny * FH + (ny + 1) * margin
        grid_w = nx * FW + (nx + 1) * margin
        img = np.zeros((grid_h, grid_w)) + 0.5 # 背景をグレーにする

        for i in range(FN):
            r = i // nx
            c = i % nx
            w = conv_layer.W[i, 0] # 最初のみchを表示
            # 正規化 (0.0 - 1.0)
            w_min, w_max = np.min(w), np.max(w)
            w = (w - w_min) / (w_max - w_min + 1e-5)
            
            y_start = r * (FH + margin) + margin
            x_start = c * (FW + margin) + margin
            img[y_start : y_start+FH, x_start : x_start+FW] = w

        plt.figure(num=title, figsize=(6, 6))
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(title)
        plt.show()

    def show_evaluation(self, model, X_test, Y_test):
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.gridspec as gridspec

        probs = model.predict(X_test)
        Y_pred = np.argmax(probs, axis=1)
        Y_true = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test

        error_indices = np.where(Y_pred != Y_true)[0]

        fig_eval = plt.figure(num="Evaluation Results", figsize=(14, 6))
        gs = gridspec.GridSpec(3, 6, figure=fig_eval)

        ax_cm = fig_eval.add_subplot(gs[:, :3])
        cm = confusion_matrix(Y_true, Y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_title("Confusion Matrix", fontsize=14)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")

        if len(error_indices) == 0:
            ax_msg = fig_eval.add_subplot(gs[:, 3:])
            ax_msg.text(0.5, 0.5, "Perfect! 0 Errors.", ha='center', va='center', fontsize=20)
            ax_msg.axis('off')
            return

        if self.input_dim == 784:
            img_size = int(np.sqrt(self.input_dim))
            num_show = min(9, len(error_indices))
            for i in range(num_show):
                idx = error_indices[i]
                row = i // 3
                col = 3 + (i % 3)
                ax_img = fig_eval.add_subplot(gs[row, col])
                ax_img.imshow(X_test[idx].reshape(img_size, img_size), cmap='gray')
                ax_img.set_title(f"T:{Y_true[idx]} -> P:{Y_pred[idx]}", color='red', fontsize=10)
                ax_img.axis('off')
        elif self.input_dim == 2:
            ax_err = fig_eval.add_subplot(gs[:, 3:])
            ax_err.scatter(X_test[:, 0], X_test[:, 1], c=Y_true, cmap='tab10', alpha=0.2)
            ax_err.scatter(X_test[error_indices, 0], X_test[error_indices, 1], color='red', marker='x', s=100)
            ax_err.set_title("Error Distribution")
        else:
            ax_txt = fig_eval.add_subplot(gs[:, 3:])
            ax_txt.axis('off')
            msg = "Misclassified Samples:\n"
            num_show = min(10, len(error_indices))
            for i in range(num_show):
                idx = error_indices[i]
                msg += f"Index {idx}: True={Y_true[idx]}, Pred={Y_pred[idx]}\n"
            ax_txt.text(0.1, 0.9, msg, va='top')

    def finish(self):
        plt.show()
