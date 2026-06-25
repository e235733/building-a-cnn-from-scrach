class LayerSpec:
    """Base class for layer specification objects."""

    def as_dict(self):
        data = self.__dict__.copy()
        if '_type' in data:
            data.pop('_type')
        return data

    @property
    def type(self):
        return getattr(self, '_type', None)


class Conv(LayerSpec):
    def __init__(self, out_channels, filter_size=3, stride=1, pad=1):
        self._type = 'Conv'
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad


class BatchNorm(LayerSpec):
    def __init__(self):
        self._type = 'BatchNorm'


class Pool(LayerSpec):
    def __init__(self, pool_h=2, pool_w=2, stride=2):
        self._type = 'Pool'
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride


class GAP(LayerSpec):
    def __init__(self):
        self._type = 'GAP'


class Affine(LayerSpec):
    def __init__(self, output_size):
        self._type = 'Affine'
        self.output_size = output_size


class Relu(LayerSpec):
    def __init__(self):
        self._type = 'Relu'


class LeakyRelu(LayerSpec):
    def __init__(self, alpha=0.01):
        self._type = 'LeakyRelu'
        self.alpha = alpha


class Dropout(LayerSpec):
    def __init__(self, dropout_ratio=0.5):
        self._type = 'Dropout'
        self.dropout_ratio = dropout_ratio


class ResidualBlock(LayerSpec):
    def __init__(self, out_channels, filter_size=3, stride=1, pad=1,
                 use_1x1_conv=None, survival_prob=1.0):
        self._type = 'ResidualBlock'
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.use_1x1_conv = use_1x1_conv
        self.survival_prob = survival_prob
