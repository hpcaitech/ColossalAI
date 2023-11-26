import torch

from colossalai.auto_parallel.tensor_shard.utils import (
    get_broadcast_shape,
    is_broadcastable,
    recover_sharding_spec_for_broadcast_shape,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec


def test_is_broadcastable():
    x1 = torch.rand(4, 4, 8)
    x2 = torch.rand(1, 8)
    assert is_broadcastable(x1.shape, x2.shape)

    x1 = torch.rand(4, 2, 8)
    x2 = torch.rand(2, 8)
    assert is_broadcastable(x1.shape, x2.shape)

    x1 = torch.rand(4, 2, 8)
    x2 = torch.rand(4, 8)
    assert not is_broadcastable(x1.shape, x2.shape)


def test_get_broadcast_shape():
    x1 = torch.rand(4, 4, 8)
    x2 = torch.rand(1, 8)
    assert get_broadcast_shape(x1.shape, x2.shape) == [4, 4, 8]

    x1 = torch.rand(4, 2, 8)
    x2 = torch.rand(2, 8)
    assert get_broadcast_shape(x1.shape, x2.shape) == [4, 2, 8]

    x1 = torch.rand(4, 2, 8)
    x2 = torch.rand(8)
    assert get_broadcast_shape(x1.shape, x2.shape) == [4, 2, 8]


def test_recover_sharding_spec_for_broadcast_shape():
    x1 = torch.rand(4, 1, 8)
    x2 = torch.rand(2, 8)

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

    broadcast_shape = get_broadcast_shape(x1.shape, x2.shape)
    logical_sharding_spec_for_x1 = ShardingSpec(
        device_mesh=device_mesh, dim_partition_dict={0: [0], 1: [1]}, entire_shape=broadcast_shape
    )
    physical_sharding_spec_for_x1, removed_dims = recover_sharding_spec_for_broadcast_shape(
        logical_sharding_spec_for_x1, broadcast_shape, x1.shape
    )
    print(physical_sharding_spec_for_x1)

    assert physical_sharding_spec_for_x1.entire_shape == x1.shape
    # dim 1 for the physical tensor is of broadcast type MULTIPLE, so should ignore
    assert physical_sharding_spec_for_x1.dim_partition_dict == {0: [0]}
    assert physical_sharding_spec_for_x1.sharding_sequence == ["S0", "R", "R"]
