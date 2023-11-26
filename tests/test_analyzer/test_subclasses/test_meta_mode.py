import pytest
import torch
import torchvision.models as tm
from packaging import version

from colossalai.testing import clear_cache_before_run, parameterize

try:
    from colossalai._analyzer._subclasses import MetaTensorMode
except:
    pass
from tests.test_analyzer.test_fx.zoo import tm_models, tmm_models


def compare_all(tensor: torch.Tensor, meta_tensor: torch.Tensor):
    assert (
        tensor.shape == meta_tensor.shape
    ), f"the shape of tensor ({tensor.shape}) and meta tensor ({meta_tensor.shape}) does not match."
    assert (
        tensor.dtype == meta_tensor.dtype
    ), f"the dtype of tensor ({tensor.dtype}) and meta tensor ({meta_tensor.dtype}) does not match."
    assert (
        tensor.stride() == meta_tensor.stride()
    ), f"the stride of tensor ({tensor.stride()}) and meta tensor ({meta_tensor.stride()}) does not match."


def run_and_compare(model):
    x = torch.rand(2, 3, 224, 224, requires_grad=True)
    x_out = model(x)
    with MetaTensorMode():
        meta_x = torch.rand(2, 3, 224, 224, requires_grad=True)
        meta_out = model(meta_x)
    compare_all(x_out, meta_out)
    x_out.sum().backward()
    meta_out.sum().backward()
    compare_all(x.grad, meta_x.grad)


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="torch version < 12")
@clear_cache_before_run()
@parameterize("m", tm_models + tmm_models)
def test_meta_mode_shape(m):
    run_and_compare(m())


if __name__ == "__main__":
    test_meta_mode_shape(tm.resnet18)
