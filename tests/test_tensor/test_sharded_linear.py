import pytest
import torch
import torch.nn.functional as F

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.nn._ops._utils import gather_forward_split_backward
from colossalai.tensor import ColoParameter, ColoTensor, ProcessGroup
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    # create mlp vars
    x = ColoTensor.from_torch_tensor(torch.rand(4, 4, 8, requires_grad=True)).cuda()
    w = ColoParameter.from_torch_tensor(torch.rand(16, 8, requires_grad=True)).cuda()
    b = ColoParameter.from_torch_tensor(torch.rand(16, requires_grad=True)).cuda()

    # run normal forward
    out = F.linear(x, w, b)

    # create mesh meta
    # the mesh is in the following topo
    # [[0, 1],
    #  [2, 3]]
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    row_id = rank // 2
    column_id = rank % 2

    # create pg
    row_process_group = None
    col_process_group = None
    row_to_ranks = {0: [0, 1], 1: [2, 3]}
    col_to_ranks = {0: [0, 2], 1: [1, 3]}

    for idx in range(2):
        # row ranks
        row_ranks = row_to_ranks[idx]
        row_pg = ProcessGroup(ranks=row_ranks, tp_degree=2)

        # col ranks
        col_ranks = col_to_ranks[idx]
        col_pg = ProcessGroup(ranks=col_ranks, tp_degree=2)

        if rank in row_ranks:
            row_process_group = row_pg

        if rank in col_ranks:
            col_process_group = col_pg

    ########################
    #  RRR x RS0 -> RRS0 #
    ########################
    # w will be transposed in F.linear
    x_replica = x.detach().clone()
    w_shard = torch.chunk(w.detach().clone(), chunks=2, dim=0)[row_id]
    b_shard = torch.chunk(b.detach().clone(), chunks=2, dim=0)[row_id]

    # adding sharding spec
    x_replica.sharding_spec = ShardingSpec(device_mesh, x.shape, dim_partition_dict={})
    w_shard.sharding_spec = ShardingSpec(device_mesh, w.shape, dim_partition_dict={0: [0]})
    b_shard.sharding_spec = ShardingSpec(device_mesh, b.shape, dim_partition_dict={0: [0]})

    # check sharding spec
    assert str(x_replica.sharding_spec.sharding_sequence) == "[R, R, R]"
    assert str(w_shard.sharding_spec.sharding_sequence) == "[S0, R]"
    assert str(b_shard.sharding_spec.sharding_sequence) == "[S0]"

    w_shard.pg_axis0 = col_process_group
    w_shard.pg_axis1 = row_process_group

    out_shard = F.linear(x_replica, w_shard, b_shard)
    assert str(out_shard.sharding_spec.sharding_sequence) == "[R, R, S0]"

    # each row only has a mini-batch
    expected_out_shard = torch.chunk(out, chunks=2, dim=2)[row_id]
    assert torch.allclose(out_shard, expected_out_shard)

    ########################
    #  S0RR x RS1 -> S0RS1 #
    ########################
    # w will be transposed in F.linear
    x_shard = torch.chunk(x.detach().clone(), chunks=2, dim=0)[row_id]
    w_shard = torch.chunk(w.detach().clone(), chunks=2, dim=0)[column_id]
    b_shard = torch.chunk(b.detach().clone(), chunks=2, dim=0)[column_id]

    # adding sharding spec
    x_shard.sharding_spec = ShardingSpec(device_mesh, x.shape, dim_partition_dict={0: [0]})
    w_shard.sharding_spec = ShardingSpec(device_mesh, w.shape, dim_partition_dict={0: [1]})
    b_shard.sharding_spec = ShardingSpec(device_mesh, b.shape, dim_partition_dict={0: [1]})

    # check sharding spec
    assert str(x_shard.sharding_spec.sharding_sequence) == "[S0, R, R]"
    assert str(w_shard.sharding_spec.sharding_sequence) == "[S1, R]"
    assert str(b_shard.sharding_spec.sharding_sequence) == "[S1]"

    w_shard.pg_axis0 = col_process_group
    w_shard.pg_axis1 = row_process_group

    out_shard = F.linear(x_shard, w_shard, b_shard)

    # each row only has a mini-batch
    expected_out_shard = torch.chunk(out, chunks=2, dim=0)[row_id]
    expected_out_shard = torch.chunk(expected_out_shard, chunks=2, dim=2)[column_id]
    assert torch.allclose(out_shard, expected_out_shard)

    ########################
    #  S0RS1 x S1R -> S0RR #
    ########################
    # w will be transposed in F.linear
    x_shard = torch.chunk(x.clone(), chunks=2, dim=0)[row_id]
    x_shard = torch.chunk(x_shard, chunks=2, dim=2)[column_id]
    w_shard = torch.chunk(w.clone(), chunks=2, dim=1)[column_id]
    b_replica = b.clone()

    # adding sharding spec
    x_shard.sharding_spec = ShardingSpec(device_mesh, x.shape, dim_partition_dict={0: [0], 2: [1]})
    w_shard.sharding_spec = ShardingSpec(device_mesh, w.shape, dim_partition_dict={1: [1]})
    b_replica.sharding_spec = ShardingSpec(device_mesh, b.shape, dim_partition_dict={})

    # check sharding spec
    assert str(x_shard.sharding_spec.sharding_sequence) == "[S0, R, S1]"
    assert str(w_shard.sharding_spec.sharding_sequence) == "[R, S1]"
    assert str(b_replica.sharding_spec.sharding_sequence) == "[R]"

    w_shard.pg_axis0 = col_process_group
    w_shard.pg_axis1 = row_process_group

    out_shard = F.linear(x_shard, w_shard, b_replica)

    # each row only has a mini-batch
    expected_out_shard = torch.chunk(out, chunks=2, dim=0)[row_id]
    assert torch.allclose(out_shard, expected_out_shard)

    ########################
    #  RRS0 x S0R -> RRR #
    ########################
    # w will be transposed in F.linear
    x_shard = torch.chunk(x.clone(), chunks=2, dim=2)[row_id]
    w_shard = torch.chunk(w.clone(), chunks=2, dim=1)[row_id]
    b_replica = b.clone()

    # adding sharding spec
    x_shard.sharding_spec = ShardingSpec(device_mesh, x.shape, dim_partition_dict={2: [0]})
    w_shard.sharding_spec = ShardingSpec(device_mesh, w.shape, dim_partition_dict={1: [0]})
    b_replica.sharding_spec = ShardingSpec(device_mesh, b.shape, dim_partition_dict={})

    # check sharding spec
    assert str(x_shard.sharding_spec.sharding_sequence) == "[R, R, S0]"
    assert str(w_shard.sharding_spec.sharding_sequence) == "[R, S0]"
    assert str(b_replica.sharding_spec.sharding_sequence) == "[R]"

    w_shard.pg_axis0 = col_process_group
    w_shard.pg_axis1 = row_process_group

    out_shard = F.linear(x_shard, w_shard, b_replica)

    # each row only has a mini-batch
    expected_out_shard = out
    assert torch.allclose(out_shard, expected_out_shard)

    ########################
    #  RS0S1 x S1R -> RS0R #
    ########################
    # w will be transposed in F.linear
    x_shard = torch.chunk(x.clone(), chunks=2, dim=1)[row_id]
    x_shard = torch.chunk(x_shard, chunks=2, dim=2)[column_id]
    w_shard = torch.chunk(w.clone(), chunks=2, dim=1)[column_id]
    b_replica = b.clone()

    # adding sharding spec
    x_shard.sharding_spec = ShardingSpec(device_mesh, x.shape, dim_partition_dict={1: [0], 2: [1]})
    w_shard.sharding_spec = ShardingSpec(device_mesh, w.shape, dim_partition_dict={1: [1]})
    b_replica.sharding_spec = ShardingSpec(device_mesh, b.shape, dim_partition_dict={})

    # check sharding spec
    assert str(x_shard.sharding_spec.sharding_sequence) == "[R, S0, S1]"
    assert str(w_shard.sharding_spec.sharding_sequence) == "[R, S1]"
    assert str(b_replica.sharding_spec.sharding_sequence) == "[R]"

    w_shard.pg_axis0 = col_process_group
    w_shard.pg_axis1 = row_process_group

    out_shard = F.linear(x_shard, w_shard, b_replica)

    # each row only has a mini-batch
    expected_out_shard = torch.chunk(out, chunks=2, dim=1)[row_id]
    assert torch.allclose(out_shard, expected_out_shard)

    ########################
    #  RRS0 x S0S1 -> RRS1 #
    ########################
    # w will be transposed in F.linear
    x_shard = torch.chunk(x.clone(), chunks=2, dim=2)[row_id]
    w_shard = torch.chunk(w.clone(), chunks=2, dim=1)[row_id]
    w_shard = torch.chunk(w_shard, chunks=2, dim=0)[column_id]
    b_shard = torch.chunk(b.clone(), chunks=2, dim=0)[column_id]

    # adding sharding spec
    x_shard.sharding_spec = ShardingSpec(device_mesh, x.shape, dim_partition_dict={2: [0]})
    w_shard.sharding_spec = ShardingSpec(device_mesh, w.shape, dim_partition_dict={0: [1], 1: [0]})
    b_shard.sharding_spec = ShardingSpec(device_mesh, b.shape, dim_partition_dict={0: [1]})

    # check sharding spec
    assert str(x_shard.sharding_spec.sharding_sequence) == "[R, R, S0]"
    assert str(w_shard.sharding_spec.sharding_sequence) == "[S1, S0]"
    assert str(b_shard.sharding_spec.sharding_sequence) == "[S1]"

    w_shard.pg_axis0 = col_process_group
    w_shard.pg_axis1 = row_process_group

    out_shard = F.linear(x_shard, w_shard, b_shard)

    # each row only has a mini-batch
    expected_out_shard = torch.chunk(out, chunks=2, dim=2)[column_id]
    assert torch.allclose(out_shard, expected_out_shard)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [4])
@rerun_if_address_is_in_use()
def test_sharded_mlp(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_sharded_mlp(4)
