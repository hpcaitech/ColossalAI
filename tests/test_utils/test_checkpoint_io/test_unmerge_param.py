import torch
from colossalai.utils.checkpoint_io.meta import ParamDistMeta
from colossalai.utils.checkpoint_io.distributed import unflatten_zero_param, gather_tp_param


def test_unflatten_zero_param_even() -> None:
    dist_metas = [ParamDistMeta(i, 4, 0, 1, zero_numel=64, zero_orig_shape=[4, 4, 4]) for i in range(4)]
    orig_tensor = torch.rand(4, 4, 4)
    tensors = list(orig_tensor.reshape(-1).chunk(4))
    unflattened_tensor = unflatten_zero_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, unflattened_tensor)


def test_unflatten_zero_param_uneven() -> None:
    dist_metas = [ParamDistMeta(i, 4, 0, 1, zero_numel=16, zero_orig_shape=[4, 4]) for i in range(1, 3)]
    orig_tensor = torch.rand(4, 4)
    tensors = [orig_tensor.reshape(-1)[:13], orig_tensor.reshape(-1)[13:]]
    unflattened_tensor = unflatten_zero_param(tensors, dist_metas)
    assert torch.equal(orig_tensor, unflattened_tensor)


if __name__ == '__main__':
    test_unflatten_zero_param_even()
    test_unflatten_zero_param_uneven()
