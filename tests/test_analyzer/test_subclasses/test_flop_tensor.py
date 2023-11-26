import pytest
import torch
import torch.nn.functional as F
import torchvision.models as tm
from packaging import version

from tests.test_analyzer.test_fx.zoo import tm_models, tmm_models

try:
    from colossalai._analyzer._subclasses import MetaTensorMode, flop_count
except:
    pass


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="torch version < 12")
@pytest.mark.parametrize("m", tm_models + tmm_models)
def test_flop_count_module(m):
    x = torch.rand(2, 3, 224, 224)
    with MetaTensorMode():  # save time for testing
        module = m()
    rs_fwd, rs_bwd = flop_count(module, x, verbose=True)
    assert rs_fwd > 0, f"fwd flop count of {m.__name__} is {rs_fwd}"
    assert rs_bwd > 0, f"bwd flop count of {m.__name__} is {rs_bwd}"


odd_cases = [
    (F.relu, (torch.rand(2, 3, 224, 224, requires_grad=True),), {"inplace": True}),
    (
        F.max_pool2d,
        (torch.rand(2, 3, 224, 224, requires_grad=True),),
        {"kernel_size": 3, "stride": 2, "padding": 1, "dilation": 2},
    ),
    (
        torch.where,
        (
            torch.rand(2, 3, 224, 224) > 0.5,
            torch.rand(2, 3, 224, 224, requires_grad=True),
            torch.rand(2, 3, 224, 224, requires_grad=True),
        ),
        {},
    ),
]


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="torch version < 12")
@pytest.mark.parametrize("func, args, kwargs", odd_cases)
def test_flop_count_function(func, args, kwargs):
    rs_fwd, rs_bwd = flop_count(func, *args, **kwargs, verbose=True)
    assert rs_fwd > 0, f"fwd flop count of {func.__name__} is {rs_fwd}"
    assert rs_bwd > 0, f"bwd flop count of {func.__name__} is {rs_bwd}"


if __name__ == "__main__":
    test_flop_count_module(tm.resnet18)
    test_flop_count_function(F.relu, (torch.rand(2, 3, 224, 224, requires_grad=True),), {"inplace": True})
