from common.layer_spec import Conv, Relu, Pool, ResidualBlock, Affine
from nn import NN_Model


def test_layer_spec_config():
    layer_config = [
        Conv(16, 5, stride=1, pad=0),
        Relu(),
        ResidualBlock(16, 3, stride=1, pad=1, survival_prob=0.9),
        Pool(2, 2, stride=2),
        Affine(10)
    ]

    model = NN_Model(input_dim=(1, 28, 28), layer_config=layer_config)
    assert len(model.layers) == 5
    assert model.layers['Conv1'].__class__.__name__ == 'Convolution'
    assert model.layers['ResidualBlock1'].__class__.__name__ == 'ResidualBlock'
    assert model.layers['Affine1'].__class__.__name__ == 'Affine'
    print('LayerSpec format works.')


if __name__ == '__main__':
    test_layer_spec_config()
