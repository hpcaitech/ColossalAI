import math
import torch
import torch.distributed as dist
import pytest
import colossalai
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _get_default_group
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.tensor import DistSpecManager, distspec, ProcessGroup
from functools import partial


def run():
    group = ProcessGroup(tp_degree=dist.get_world_size())
    rank = dist.get_rank()
    size = dist.get_world_size()
    depth = int(math.sqrt(size))
    assert depth == math.sqrt(size)
    x = torch.rand(8, 8).cuda()
    old_dist_spec = distspec.replicate()
    row_spec = distspec.shard(group, [0], [size])
    col_spec = distspec.shard(group, [-1], [size])
    mat_spec = distspec.shard(group, [0, 1], [depth, depth])
    row_shard = DistSpecManager._shard_as(x, old_dist_spec, row_spec)
    assert torch.equal(x.chunk(size, 0)[rank], row_shard)
    assert torch.equal(x, DistSpecManager._gather(row_shard, row_spec))
    col_shard = DistSpecManager._all_to_all(row_shard, row_spec, col_spec)
    assert torch.equal(x.chunk(size, -1)[rank], col_shard)
    assert torch.equal(x, DistSpecManager._gather(col_shard, col_spec))
    mat_shard = DistSpecManager._shard_as(x, old_dist_spec, mat_spec)
    assert torch.equal(x.chunk(depth, 0)[rank // depth].chunk(depth, 1)[rank % depth], mat_shard)
    assert torch.equal(x, DistSpecManager._gather(mat_shard, mat_spec))


def check_mem():
    group = ProcessGroup(tp_degree=dist.get_world_size())
    size = dist.get_world_size()
    assert torch.cuda.memory_allocated() == 0
    x = torch.rand(32, 32).cuda()
    orig_mem = x.numel() * x.element_size()
    assert torch.cuda.memory_allocated() == orig_mem
    old_dist_spec = distspec.replicate()
    row_spec = distspec.shard(group, [0], [size])
    x.data = DistSpecManager._shard_as(x, old_dist_spec, row_spec)
    assert x.size(0) == 32 // size and x.size(1) == 32
    assert torch.cuda.memory_allocated() == orig_mem // size
    x.data = DistSpecManager._gather(x, row_spec)
    assert torch.cuda.memory_allocated() == orig_mem


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_mem()
    run()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_dist_spec_mgr(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_dist_spec_mgr(4)
