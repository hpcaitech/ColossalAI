import torch
import torch.nn


def test_maxpool():
    layer_to_test = dict(maxpool_1d=dict(layer=torch.nn.MaxPool1d, shape=(4, 3, 4)),
                         maxpool_2d=dict(layer=torch.nn.MaxPool2d, shape=(4, 3, 4, 4)))

    for name, info in layer_to_test.items():
        data = torch.rand(*info['shape'])
        meta_data = data.to('meta')
        layer = info['layer'](kernel_size=3)
        out = layer(data)
        meta_out = layer(meta_data)
        assert meta_out.is_meta
        assert out.shape == meta_out.shape


def test_avgpool():
    layer_to_test = dict(maxpool_1d=dict(layer=torch.nn.AvgPool1d, shape=(4, 3, 4)),
                         maxpool_2d=dict(layer=torch.nn.AvgPool2d, shape=(4, 3, 4, 4)),
                         maxpool_3d=dict(layer=torch.nn.AvgPool3d, shape=(4, 3, 4, 4, 4)))

    for name, info in layer_to_test.items():
        data = torch.rand(*info['shape'])
        meta_data = data.to('meta')
        layer = info['layer'](kernel_size=3)
        out = layer(data)
        meta_out = layer(meta_data)
        assert meta_out.is_meta
        assert out.shape == meta_out.shape
