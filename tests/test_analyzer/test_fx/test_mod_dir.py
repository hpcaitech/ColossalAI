import pytest
import torch

from colossalai.testing import clear_cache_before_run, parameterize

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
        self.conv = torch.nn.Conv2d(
            in_channel, out_channels, kernel_size, bias=bias, padding=1, stride=2, dilation=2, groups=3
        )
        self.conv_transpose = torch.nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size, bias=bias, padding=1, stride=2, dilation=2, groups=3
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_transpose(x)
        return x


class AModel(torch.nn.Module):
    def __init__(self, bias) -> None:
        super().__init__()
        self.linear_1 = LinearModel(3, 3, bias)
        self.linear_2 = LinearModel(3, 3, bias)
        self.conv = ConvModel(3, 6, 3, bias)

    def forward(self, x):
        for i in range(x.shape[0]):
            x = self.linear_1(x)
            x = self.linear_2(x)
        x = self.conv(x)
        return x


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="torch version < 12")
@clear_cache_before_run()
@parameterize("bias", [True, False])
@parameterize("bias_addition_split", [True, False])
@parameterize("shape", [(3, 3, 3), (3, 3, 3, 3)])
def test_mod_dir(bias, bias_addition_split, shape):
    model = AModel(bias=bias)
    x = torch.rand(shape)
    gm = symbolic_trace(model, meta_args={"x": x}, bias_addition_split=bias_addition_split)
    for node in gm.graph.nodes:
        assert len(node.meta["info"].mod_dir), f"{node} should have non-trivial ``mod_dir``."
        print(node, node.meta["info"].mod_dir)


if __name__ == "__main__":
    test_mod_dir(bias=True, bias_addition_split=True, shape=(3, 3, 3))
