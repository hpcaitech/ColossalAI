from typing import Any, Callable, Union

import pytest
import torch
import torch.nn as nn

from colossalai.testing import clear_cache_before_run

try:
    from colossalai._analyzer._subclasses import MetaTensor
except:
    pass

aten = torch.ops.aten

registered_meta = {
    ("aten.convolution.default", True): [  # (aten ops, requires_backward)
        (nn.Conv1d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2), torch.rand(2, 3, 4)),
        (nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2), torch.rand(2, 3, 4, 4)),
        (nn.Conv3d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2), torch.rand(2, 3, 4, 4, 4)),
        (nn.ConvTranspose1d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2), torch.rand(2, 3, 4)),
        (
            nn.ConvTranspose2d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2),
            torch.rand(2, 3, 4, 4),
        ),
        (
            nn.ConvTranspose3d(in_channels=3, out_channels=4, kernel_size=2, padding=1, dilation=2),
            torch.rand(2, 3, 4, 4, 4),
        ),
    ],
    ("aten.native_batch_norm.default", True): [
        (nn.BatchNorm1d(4), torch.rand(2, 4)),
        (nn.BatchNorm2d(4), torch.rand(1, 4, 4, 4)),
        (nn.BatchNorm3d(4), torch.rand(1, 4, 4, 4, 4)),
    ],
    ("aten.native_layer_norm.default", True): [
        (nn.LayerNorm(4), torch.rand(1, 2, 3, 4)),
    ],
    ("aten.avg_pool1d.default", True): [
        (nn.MaxPool1d(3, stride=2), torch.rand(4, 5, 5)),
        (nn.AvgPool1d(3, stride=2), torch.rand(4, 5, 5)),
        (nn.AdaptiveMaxPool1d(3), torch.rand(4, 5, 5)),
        (nn.AdaptiveAvgPool1d(3), torch.rand(4, 5, 5)),
    ],
    ("aten.avg_pool2d.default", True): [
        (nn.MaxPool2d((3, 2), stride=(2, 1)), torch.rand(2, 4, 5, 5)),
        (nn.AvgPool2d((3, 2), stride=(2, 1)), torch.rand(2, 4, 5, 5)),
        (nn.AdaptiveMaxPool2d((3, 2)), torch.rand(2, 4, 5, 5)),
        (nn.AdaptiveAvgPool2d((3, 2)), torch.rand(2, 4, 5, 5)),
    ],
    ("aten.relu.default", True): [
        (nn.ReLU(), torch.rand(4, 3, 1, 2)),
        (nn.LeakyReLU(), torch.rand(4, 3, 1, 2)),
        (nn.SiLU(), torch.rand(4, 3, 1, 2)),
        (nn.GELU(), torch.rand(4, 3, 1, 2)),
        (nn.ELU(), torch.rand(4, 3, 1, 2)),
        (nn.Sigmoid(), torch.rand(4, 3, 1, 2)),
        (nn.Tanh(), torch.rand(4, 3, 1, 2)),
        (nn.Hardswish(), torch.rand(4, 3, 1, 2)),
    ],
}


def compare_all(tensor: torch.Tensor, meta_tensor: torch.Tensor) -> Any:
    assert (
        tensor.shape == meta_tensor.shape
    ), f"the shape of tensor ({tensor.shape}) and meta tensor ({meta_tensor.shape}) does not match."
    assert (
        tensor.dtype == meta_tensor.dtype
    ), f"the dtype of tensor ({tensor.dtype}) and meta tensor ({meta_tensor.dtype}) does not match."
    assert (
        tensor.stride() == meta_tensor.stride()
    ), f"the stride of tensor ({tensor.stride()}) and meta tensor ({meta_tensor.stride()}) does not match."


def run_and_compare(f: Union[nn.Module, Callable], x: torch.Tensor, requires_backward=False) -> Any:
    x.requires_grad = requires_backward
    meta_x = MetaTensor(x)
    x_out, meta_out = f(x), f(meta_x)
    compare_all(x_out, meta_out)
    if requires_backward:
        x_out.sum().backward()
        meta_out.sum().backward()
        compare_all(x.grad, meta_x.grad)


@pytest.mark.skipif(torch.__version__ < "1.12.0", reason="torch version < 12")
@clear_cache_before_run()
def test_meta_aten():
    for (aten_op, requires_backward), v in registered_meta.items():
        for f, x in v:
            run_and_compare(f, x, requires_backward)


if __name__ == "__main__":
    test_meta_aten()
