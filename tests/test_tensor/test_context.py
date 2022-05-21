import pytest
from colossalai.utils import ColoInitContext

from numpy import allclose, require
import torch
from colossalai.tensor import ColoTensor
from copy import deepcopy

from colossalai.utils.cuda import get_current_device


@pytest.mark.skip
# FIXME(ver217): support lazy init
def test_lazy_init():
    in_dim = 4
    out_dim = 5

    with ColoInitContext(lazy_memory_allocate=True) as ctx:
        fc = torch.nn.Linear(in_dim, out_dim, bias=True)

    # lazy_memory_allocate=True, no payload is maintained
    assert fc.weight._torch_tensor.numel() == 0

    fc.weight.torch_tensor()
    assert fc.weight._torch_tensor.numel() == in_dim * out_dim


@pytest.mark.skip
def test_device():
    in_dim = 4
    out_dim = 5

    with ColoInitContext(lazy_memory_allocate=True, device=get_current_device()) as ctx:
        fc = torch.nn.Linear(in_dim, out_dim, bias=True)

    # eval an lazy parameter
    fc.weight.torch_tensor()
    assert fc.weight.device == get_current_device()


if __name__ == '__main__':
    test_lazy_init()
    test_device()
