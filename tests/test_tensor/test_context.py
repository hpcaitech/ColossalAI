from colossalai.utils import ColoInitContext

from numpy import allclose, require
import torch
from colossalai.tensor import ColoTensor
from copy import deepcopy


def test_linear():
    in_dim = 4
    out_dim = 5

    with ColoInitContext(lazy_memory_allocate=True) as ctx:
        fc = torch.nn.Linear(in_dim, out_dim, bias=True)

    print(fc.weight.numel())
    print(fc.bias.numel())

    # lazy_memory_allocate=True, no payload is maintained
    assert fc.weight._torch_tensor.numel() == 0

    fc.weight.torch_tensor()
    assert fc.weight._torch_tensor.numel() == in_dim * out_dim


if __name__ == '__main__':
    test_linear()
