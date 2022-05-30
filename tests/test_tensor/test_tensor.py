import torch
import pytest
from colossalai.tensor import ColoTensor
from numpy import allclose


def test_tensor_indexing():
    torch_t = torch.randn(2, 3)
    colo_t = ColoTensor(torch_t)
    assert allclose(torch_t[:, 1], colo_t[:, 1])


@pytest.mark.skip
# FIXME(ver217): support lazy init
def test_lazy_init_tensor():
    lazy_t = ColoTensor(2, 3, dtype=torch.float32, requires_grad=True)
    assert lazy_t._torch_tensor.numel() == 0
    assert lazy_t.numel() == 6 == lazy_t.torch_tensor().numel()


def test_wrapped_tensor_func():
    t_ref = torch.randn(4, 5)
    t = ColoTensor.from_torch_tensor(t_ref.clone())

    # non-func attr
    assert t.is_cuda == t_ref.is_cuda

    # TODO I don't find out a tensor function which returns None.

    # return 1 torch.Tensor
    t_abs = t.abs()
    assert isinstance(t_abs, ColoTensor) and torch.equal(t_abs, t_ref.abs())

    # return 1 non-torch.Tensor
    assert t.dim() == t_ref.dim()

    # return >1 torch.Tensor
    t_split1, t_split2 = t.split(2)
    assert isinstance(t_split1, ColoTensor) and isinstance(t_split2, ColoTensor)


def test_operand():
    t_ref = torch.randn(4, 5)
    t = ColoTensor.from_torch_tensor(t_ref.clone())

    t_ref_res = t_ref + t_ref
    t_res = t + t
    assert torch.allclose(t_ref_res, t_res)

