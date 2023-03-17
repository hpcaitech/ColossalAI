import pytest
import torch
from torch.utils.checkpoint import checkpoint

try:
    from colossalai._analyzer.fx import symbolic_trace
except:
    pass


class LinearModel(torch.nn.Module):

    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvModel(torch.nn.Module):

    def __init__(self, in_channel, out_channels, kernel_size, bias) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channel,
                                    out_channels,
                                    kernel_size,
                                    bias=bias,
                                    padding=1,
                                    stride=2,
                                    dilation=2,
                                    groups=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channel,
                                                       out_channels,
                                                       kernel_size,
                                                       bias=bias,
                                                       padding=1,
                                                       stride=2,
                                                       dilation=2,
                                                       groups=3)

    def forward(self, x, select=0):
        if select == 0:
            x = self.conv(x)
        else:
            x = self.conv_transpose(x)
        return x


class SiuModel(torch.nn.Module):

    def __init__(self, bias) -> None:
        super().__init__()
        self.linear = LinearModel(3, 3, bias)
        self.conv = ConvModel(3, 6, 3, bias)

    def forward(self, x, select=0):
        x = self.linear(x)
        x = checkpoint(self.conv, x, select)
        return x


class AddmmModel(torch.nn.Module):

    def __init__(self, alpha, beta) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        x = torch.addmm(x, x, x, alpha=self.alpha, beta=self.beta)
        return x


@pytest.mark.skipif(torch.__version__ < '1.12.0', reason='torch version < 12')
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("bias_addition_split", [True, False])
@pytest.mark.parametrize("shape", [(3, 3, 3), (3, 3, 3, 3)])
@pytest.mark.parametrize("select", [0, 1])
def test_siu_model(bias, bias_addition_split, shape, select):
    model = SiuModel(bias=bias)
    x = torch.rand(shape)
    gm = symbolic_trace(model,
                        meta_args={'x': x},
                        concrete_args={'select': select},
                        trace_act_ckpt=True,
                        bias_addition_split=bias_addition_split)
    assert torch.allclose(model(x, select), gm(x, select)), 'original model and traced model should be the same!'
    if bias and bias_addition_split:
        assert '+' in gm.code, 'bias addition should be split!'
    else:
        assert '+' not in gm.code, 'bias addition should not be split!'


@pytest.mark.skipif(torch.__version__ < '1.12.0', reason='torch version < 12')
@pytest.mark.parametrize("alpha", [1, 2])
@pytest.mark.parametrize("beta", [1, 2])
@pytest.mark.parametrize("bias_addition_split", [True, False])
@pytest.mark.parametrize("shape", [(3, 3), (5, 5)])
def test_addmm_model(alpha, beta, bias_addition_split, shape):
    model = AddmmModel(alpha=alpha, beta=beta)
    x = torch.rand(shape)
    gm = symbolic_trace(model, meta_args={'x': x}, trace_act_ckpt=True, bias_addition_split=bias_addition_split)
    assert torch.allclose(model(x), gm(x)), 'original model and traced model should be the same!'
    if (alpha == 1 and beta == 1) or not bias_addition_split:
        assert '*' not in gm.code, 'bias addition should not be split!'
    elif bias_addition_split:
        assert '+' in gm.code, 'bias addition should be split!'


if __name__ == '__main__':
    test_siu_model(True, True, (3, 3, 3))
