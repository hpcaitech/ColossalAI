from colossalai.tensor.shape_consistency import ShapeConsistencyManager
import torch
from colossalai.tensor.sharding_spec import _DimSpec, ShardingSpec
from colossalai.device.device_mesh import DeviceMesh


def test_shape_consistency():
    physical_mesh_id = torch.arange(0, 16).reshape(2, 8)
    mesh_shape = (4, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    #  [8, 9, 10,11],
    #  [12,13,14,15]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((4, 8, 6))
    dim_partition_dict = {0: [0], 1: [1]}
    # DistSpec:
    #     shard_sequence: S0,S1,R
    #     device_mesh_shape: (4, 4)
    sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)
    shape_consistency_manager = ShapeConsistencyManager()
    # {DistSpec:
    #     shard_sequence: R,S1,R
    #     device_mesh_shape: (4, 4): 0, DistSpec:
    #     shard_sequence: S0,R,R
    #     device_mesh_shape: (4, 4): 0}
    rst_dict = shape_consistency_manager.get_all_all_gather_spec(sharding_spec, 0)

    assert '[R, S1, R]' in [
        str(all_gather_sharding_spec.sharding_sequence) for all_gather_sharding_spec in rst_dict.keys()
    ]
    assert '[S0, R, R]' in [
        str(all_gather_sharding_spec.sharding_sequence) for all_gather_sharding_spec in rst_dict.keys()
    ]

    dim_partition_dict_all2all = {0: [0], 1: [1]}
    # DistSpec:
    #     shard_sequence: S0,S1,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_all2all = ShardingSpec(device_mesh, entire_shape, dim_partition_dict_all2all)
    # {DistSpec:
    #     shard_sequence: S01,R,R
    #     device_mesh_shape: (4, 4): 0, DistSpec:
    #     shard_sequence: R,S1,S0
    #     device_mesh_shape: (4, 4): 0, DistSpec:
    #     shard_sequence: S0,R,S1
    #     device_mesh_shape: (4, 4): 0}
    rst_dict_all2all = shape_consistency_manager.get_all_all_to_all_spec(sharding_spec_all2all, 0)

    assert '[S01, R, R]' in [
        str(all2all_sharding_spec.sharding_sequence) for all2all_sharding_spec in rst_dict_all2all.keys()
    ]
    assert '[R, S1, S0]' in [
        str(all2all_sharding_spec.sharding_sequence) for all2all_sharding_spec in rst_dict_all2all.keys()
    ]
    assert '[S0, R, S1]' in [
        str(all2all_sharding_spec.sharding_sequence) for all2all_sharding_spec in rst_dict_all2all.keys()
    ]

    dim_partition_shard = {0: [0]}
    # DistSpec:
    #     shard_sequence: S0,R,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_shard = ShardingSpec(device_mesh, entire_shape, dim_partition_shard)
    # {DistSpec:
    #     shard_sequence: S01,R,R
    #     device_mesh_shape: (4, 4): 0, DistSpec:
    #     shard_sequence: S0,S1,R
    #     device_mesh_shape: (4, 4): 0, DistSpec:
    #     shard_sequence: S0,R,S1
    #     device_mesh_shape: (4, 4): 0}
    rst_dict_shard = shape_consistency_manager.get_all_shard_spec(sharding_spec_shard, 0)

    assert '[S01, R, R]' in [
        str(shard_sharding_spec.sharding_sequence) for shard_sharding_spec in rst_dict_shard.keys()
    ]
    assert '[S0, S1, R]' in [
        str(shard_sharding_spec.sharding_sequence) for shard_sharding_spec in rst_dict_shard.keys()
    ]
    assert '[S0, R, S1]' in [
        str(shard_sharding_spec.sharding_sequence) for shard_sharding_spec in rst_dict_shard.keys()
    ]


if __name__ == '__main__':
    test_shape_consistency()
