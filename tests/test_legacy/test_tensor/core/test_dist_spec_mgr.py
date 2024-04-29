import math

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.legacy.tensor import DistSpecManager, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn


def run():
    group = ProcessGroup(tp_degree=dist.get_world_size())
    rank = dist.get_rank()
    size = dist.get_world_size()
    depth = int(math.sqrt(size))
    assert depth == math.sqrt(size)
    x = torch.rand(8, 8).cuda()
    old_dist_spec = ReplicaSpec()
    row_spec = ShardSpec([0], [size])
    col_spec = ShardSpec([-1], [size])
    mat_spec = ShardSpec([0, 1], [depth, depth])
    row_shard = DistSpecManager._shard_as(x, old_dist_spec, row_spec, group)
    assert torch.equal(x.chunk(size, 0)[rank], row_shard)
    assert torch.equal(x, DistSpecManager._gather(row_shard, row_spec, group))
    col_shard = DistSpecManager._all_to_all(row_shard, row_spec, col_spec, group)
    assert torch.equal(x.chunk(size, -1)[rank], col_shard)
    assert torch.equal(x, DistSpecManager._gather(col_shard, col_spec, group))
    mat_shard = DistSpecManager._shard_as(x, old_dist_spec, mat_spec, group)
    assert torch.equal(x.chunk(depth, 0)[rank // depth].chunk(depth, 1)[rank % depth], mat_shard)
    assert torch.equal(x, DistSpecManager._gather(mat_shard, mat_spec, group))


def check_mem():
    pg = ProcessGroup(tp_degree=dist.get_world_size())
    size = dist.get_world_size()
    assert torch.cuda.memory_allocated() == 0
    x = torch.rand(32, 32).cuda()
    orig_mem = x.numel() * x.element_size()
    assert torch.cuda.memory_allocated() == orig_mem
    old_dist_spec = ReplicaSpec()
    row_spec = ShardSpec([0], [size])
    x.data = DistSpecManager._shard_as(x, old_dist_spec, row_spec, pg)
    assert x.size(0) == 32 // size and x.size(1) == 32
    assert torch.cuda.memory_allocated() == orig_mem // size
    x.data = DistSpecManager._gather(x, row_spec, pg)
    assert torch.cuda.memory_allocated() == orig_mem


def run_dist(rank, world_size, port):
    colossalai.legacy.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    check_mem()
    run()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 4])
@rerun_if_address_is_in_use()
def test_dist_spec_mgr(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_dist_spec_mgr(4)
