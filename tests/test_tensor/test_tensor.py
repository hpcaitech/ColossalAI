import torch
import pytest
from colossalai.tensor import ColoTensor
from numpy import allclose

import colossalai
from colossalai.utils import free_port
from colossalai.tensor import distspec, TensorSpec
from colossalai.core import global_context as gpc
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.tensor import distspec, TensorSpec, ColoTensor
from colossalai.context import ParallelMode
from functools import partial


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


#### Test Distributed init a Colotensor


def _run_view(world_size):
    t_ref = torch.randn(4, 5)
    t = ColoTensor.from_torch_tensor(
        t_ref,
        TensorSpec(distspec.shard(process_group=gpc.get_group(ParallelMode.DATA), dims=[0],
                                  num_partitions=[world_size])))

    assert t.size_global()[0] == 4 * world_size
    assert t.size_global(1) == 5
    assert t.size_global() == torch.Size([4 * world_size, 5])

    t.view_local(4 * 5)
    assert t.tensor_spec.dist_spec.placement.value == 's'

    t = t.view_global(4 * 5 * world_size)
    assert t.tensor_spec.dist_spec.placement.value == 'r'
    assert t.shape == torch.Size([4 * 5 * world_size])


def _run_tensor_shard_init(world_size):
    t_ref = torch.randn(4, 5)
    print(gpc.get_group(ParallelMode.DATA).size())
    shard_spec = distspec.shard(process_group=gpc.get_group(ParallelMode.DATA), dims=[0], num_partitions=[world_size])
    tensor_spec = TensorSpec(shard_spec)
    t = ColoTensor.from_torch_tensor(t_ref.clone(), tensor_spec)
    t.set_tensor_spec(TensorSpec(dist_spec=distspec.replicate()))
    assert t.shape == torch.Size((4 * world_size, 5))


def _run_tensor_replicated_init(world_size):
    t_ref = torch.randn(4 * world_size, 5)
    t = ColoTensor.from_torch_tensor(t_ref.clone())

    assert t.shape == torch.Size((4 * world_size, 5)), f"{t.shape}"


def run_dist_tests(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_tensor_shard_init(world_size)
    _run_tensor_replicated_init(world_size)
    _run_view(world_size)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_dist_cases(world_size):
    run_func = partial(run_dist_tests, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_dist_cases(2)
