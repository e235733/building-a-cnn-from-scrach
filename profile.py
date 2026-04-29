import cProfile as profile
from mnist_dataset import MnistDataset
from cnn import CNN_Model
from common.trainer import Trainer

def profile_training():
    # Small dataset for profiling
    n_samples = 1000
    epochs = 1
    mini_batch_size = 100

    dataset = MnistDataset(n_samples=n_samples)
    x_train = dataset.X_train.reshape(-1, 1, 28, 28)
    x_test = dataset.X_test.reshape(-1, 1, 28, 28)
    t_train = dataset.Y_train
    t_test = dataset.Y_test

    model = CNN_Model(
        input_dim=(1, 28, 28),
        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 2, 'stride': 1},
        hidden_size=100, 
        output_size=10, 
        weight_init_std=0.01
    )

    trainer = Trainer(
        model, x_train, t_train, x_test, t_test,
        epochs=epochs, 
        mini_batch_size=mini_batch_size,
        optimizer='SGD', 
        optimizer_param={'lr':0.01},
        evaluate_sample_num_per_epoch=None,
        verbose=False
    )

    trainer.train()

if __name__ == '__main__':
    profile.run('profile_training()', 'profile_output.prof')