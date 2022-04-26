import torch
from colossalai.tensor import ColoTensor
from numpy import allclose


def test_tensor_indexing():
    torch_t = torch.randn(2, 3)
    colo_t = ColoTensor.init_from_torch_tensor(torch_t)
    assert allclose(torch_t[:, 1], colo_t[:, 1].torch_tensor())


def test_lazy_init_tensor():
    lazy_t = ColoTensor(2, 3, dtype=torch.float32, requires_grad=True)
    assert lazy_t._torch_tensor.numel() == 0
    assert lazy_t.numel() == 6 == lazy_t.torch_tensor().numel()
