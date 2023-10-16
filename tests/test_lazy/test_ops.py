import copy

import pytest
import torch
import torch.nn as nn
from lazy_init_utils import SUPPORT_LAZY
from torch.nn import Parameter

from colossalai.lazy import LazyInitContext


@pytest.mark.skipif(not SUPPORT_LAZY, reason="requires torch >= 1.12.0")
def test_lazy_ops():
    with LazyInitContext():
        x = torch.rand(2, 3)
        assert tuple(x.shape) == (2, 3)
        assert x.device.type == "cpu"
        x.requires_grad is False
        y = x.cuda()
        assert tuple(y.shape) == (2, 3)
        assert y.device.type == "cuda"
        assert y.requires_grad is False
        assert x.cpu() is x
        p = Parameter(torch.empty(2, 3))
        assert tuple(p.shape) == (2, 3)
        assert p.device.type == "cpu"
        assert p.requires_grad is True
        assert isinstance(p, Parameter)
    x.materialize()
    assert tuple(x.shape) == (2, 3)
    assert x.device.type == "cpu"
    assert x.requires_grad is False
    y.materialize()
    assert tuple(y.shape) == (2, 3)
    assert y.device.type == "cuda"
    assert y.requires_grad is False
    p.materialize()
    assert tuple(p.shape) == (2, 3)
    assert p.device.type == "cpu"
    assert p.requires_grad is True
    assert isinstance(p, Parameter)

    with LazyInitContext():
        x = torch.empty(2, 3)
        x.uniform_()
    x.materialize()
    assert tuple(x.shape) == (2, 3)

    with LazyInitContext():
        model = nn.Linear(3, 4)
        model = model.cuda()
        model_copied = copy.deepcopy(model)
    LazyInitContext.materialize(model)
    assert model.weight.device.type == "cuda"
    assert model.bias.device.type == "cuda"
    LazyInitContext.materialize(model_copied)
    assert model_copied.weight.device.type == "cuda"
    assert model_copied.bias.device.type == "cuda"
    assert torch.equal(model.weight, model_copied.weight)
    assert torch.equal(model.bias, model_copied.bias)


if __name__ == "__main__":
    test_lazy_ops()
