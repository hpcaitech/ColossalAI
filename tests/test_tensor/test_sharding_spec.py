import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec


def test_sharding_spec():
    physical_mesh_id = torch.arange(0, 16)
    mesh_shape = (4, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    #  [8, 9, 10,11],
    #  [12,13,14,15]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((16, 8, 6))
    dim_partition_dict = {0: [0, 1]}
    # DistSpec:
    #     shard_sequence: S01,R,R
    #     device_mesh_shape: (4, 4)
    sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
    assert str(sharding_spec.sharding_sequence) == "[S01, R, R]"


if __name__ == '__main__':
    test_sharding_spec()
