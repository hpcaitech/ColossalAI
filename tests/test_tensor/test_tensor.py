import torch
import pytest
from colossalai.tensor import ColoTensor
from numpy import allclose

import colossalai
from colossalai.utils import free_port
from colossalai.tensor import distspec, ColoTensorSpec
from colossalai.core import global_context as gpc
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.tensor import distspec, ColoTensor, ProcessGroup, ShardSpec, ReplicaSpec
from functools import partial


def _run_tensor_indexing():
    pg = ProcessGroup()
    torch_t = torch.randn(2, 3)
    colo_t = ColoTensor(torch_t, ColoTensorSpec(pg))
    assert allclose(torch_t[:, 1], colo_t[:, 1])


def _run_wrapped_tensor_func():
    pg = ProcessGroup()
    t_ref = torch.randn(4, 5)
    t = ColoTensor.from_torch_tensor(t_ref.clone(), ColoTensorSpec(pg))

    # non-func attr
    assert t.is_cuda == t_ref.is_cuda

    # return 1 torch.Tensor
    t_abs = t.abs()
    assert isinstance(t_abs, ColoTensor) and torch.equal(t_abs, t_ref.abs())

    # return 1 non-torch.Tensor
    assert t.dim() == t_ref.dim()

    # return >1 torch.Tensor
    assert isinstance(t, ColoTensor)
    t_split1, t_split2 = t.split(2)
    assert isinstance(t_split1, ColoTensor) and isinstance(t_split2, ColoTensor), f"{type(t_split1)} {type(t_split2)}"


def _run_operand(world_size):
    pg = ProcessGroup()
    t_ref = torch.randn(4, 5)
    t = ColoTensor.from_torch_tensor(t_ref.clone(), ColoTensorSpec(pg))

    t_ref_res = t_ref + t_ref
    t_res = t + t

    assert isinstance(t_res, ColoTensor)
    assert torch.allclose(t_ref_res, t_res)

    pg = ProcessGroup(tp_degree=world_size)
    t = ColoTensor.from_torch_tensor(t_ref.clone(), ColoTensorSpec(pg))
    t.set_dist_spec(ShardSpec([0], [world_size]))
    t_new = torch.zeros_like(t)
    assert isinstance(t_new, ColoTensor)
    assert t_new.is_sharded()


#### Test Distributed init a Colotensor


def _run_view(world_size):
    t_ref = torch.randn(4, 5)
    rank = gpc.get_global_rank()
    pg = ProcessGroup(rank, list(range(world_size)), tp_degree=world_size)
    t = ColoTensor.from_torch_tensor(
        t_ref, ColoTensorSpec(pg, dist_attr=ShardSpec(dims=[0], num_partitions=[pg.tp_world_size()])))

    assert t.size_global()[0] == 4 * world_size
    assert t.size_global(1) == 5
    assert t.size_global() == torch.Size([4 * world_size, 5])

    t = t.view_global(4 * 5 * world_size)
    assert t.shape == torch.Size([4 * 5 * world_size])


def _run_tensor_shard_init(world_size):
    t_ref = torch.randn(4, 5)
    pg = ProcessGroup(tp_degree=world_size)
    shard_attr = ShardSpec(dims=[0], num_partitions=[pg.tp_world_size()])
    tensor_spec = ColoTensorSpec(pg, dist_attr=shard_attr)
    t = ColoTensor.from_torch_tensor(t_ref.clone(), tensor_spec)
    t.set_dist_spec(distspec.replicate())

    assert t.shape == torch.Size((4 * world_size, 5)), f"{t.shape} vs ({4 * world_size, 5})"


def _run_tensor_replicated_init(world_size):
    t_ref = torch.randn(4 * world_size, 5)
    pg = ProcessGroup()
    spec = ColoTensorSpec(pg)
    t = ColoTensor.from_torch_tensor(t_ref.clone(), spec)

    assert t.shape == torch.Size((4 * world_size, 5)), f"{t.shape}"


def _run_process_group(world_size):
    pg1 = ProcessGroup()
    pg2 = ProcessGroup()

    assert pg1 == pg2


def run_dist_tests(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    _run_tensor_shard_init(world_size)
    _run_tensor_replicated_init(world_size)
    _run_view(world_size)
    _run_process_group(world_size)
    _run_tensor_indexing()
    _run_operand(world_size)
    _run_wrapped_tensor_func()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 2])
@rerun_if_address_is_in_use()
def test_dist_cases(world_size):
    run_func = partial(run_dist_tests, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_dist_cases(1)
