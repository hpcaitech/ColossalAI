import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import CollectiveCommPattern, ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec

physical_mesh_id = torch.arange(0, 16)
mesh_shape = (4, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10,11],
#  [12,13,14,15]]
device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
entire_shape = torch.Size((64, 32, 16))
shape_consistency_manager = ShapeConsistencyManager()


def test_one_step_transform():

    dim_partition_dict = {0: [0], 1: [1]}
    # DistSpec:
    #     shard_sequence: S0,S1,R
    #     device_mesh_shape: (4, 4)
    sharding_spec = ShardingSpec(device_mesh, entire_shape, dim_partition_dict)

    # {DistSpec:
    #     shard_sequence: R,S1,R
    #     device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:allgather, gather_dim:0, logical_process_axis:0), 0), DistSpec:
    #     shard_sequence: S0,R,R
    #     device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:1), 0)}
    rst_dict = shape_consistency_manager.get_all_all_gather_spec(sharding_spec, {
        "forward": 0,
        "backward": 0,
        "total": 0
    })

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
    #         shard_sequence: S01,R,R
    #         device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:all2all, gather_dim:1, shard_dim:0, logical_process_axis: 1), 0), DistSpec:
    #         shard_sequence: R,S1,S0
    #         device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:all2all, gather_dim:0, shard_dim:2, logical_process_axis: 0), 0), DistSpec:
    #         shard_sequence: S0,R,S1
    #         device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:all2all, gather_dim:1, shard_dim:2, logical_process_axis: 1), 0)}
    rst_dict_all2all = shape_consistency_manager.get_all_all_to_all_spec(sharding_spec_all2all, {
        "forward": 0,
        "backward": 0,
        "total": 0
    })

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
    #         shard_sequence: S01,R,R
    #         device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:shard, shard_dim:0, logical_process_axis:1), 0), DistSpec:
    #         shard_sequence: S0,S1,R
    #         device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:shard, shard_dim:1, logical_process_axis:1), 0), DistSpec:
    #         shard_sequence: S0,R,S1
    #         device_mesh_shape: (4, 4): (CommSpec:(comm_pattern:shard, shard_dim:2, logical_process_axis:1), 0)}
    rst_dict_shard = shape_consistency_manager.get_all_shard_spec(sharding_spec_shard, {
        "forward": 0,
        "backward": 0,
        "total": 0
    })

    assert '[S01, R, R]' in [
        str(shard_sharding_spec.sharding_sequence) for shard_sharding_spec in rst_dict_shard.keys()
    ]
    assert '[S0, S1, R]' in [
        str(shard_sharding_spec.sharding_sequence) for shard_sharding_spec in rst_dict_shard.keys()
    ]
    assert '[S0, R, S1]' in [
        str(shard_sharding_spec.sharding_sequence) for shard_sharding_spec in rst_dict_shard.keys()
    ]


def test_shape_consistency():
    dim_partition_source = {1: [0, 1]}
    dim_partition_target = {0: [0, 1]}

    # DistSpec:
    #     shard_sequence: R,S01,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_source = ShardingSpec(device_mesh, entire_shape, dim_partition_source)

    # DistSpec:
    #     shard_sequence: S01,R,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_target = ShardingSpec(device_mesh, entire_shape, dim_partition_target)

    transform_path, comm_action_sequence, total_cost = shape_consistency_manager.shape_consistency(
        sharding_spec_source, sharding_spec_target)

    transform_path_str = '->'.join([str(sharding_spec.sharding_sequence) for sharding_spec in transform_path])
    assert transform_path_str == '[R, S01, R]->[R, S0, R]->[S0, R, R]->[S01, R, R]'

    # all-gather(S01) -> S0
    assert comm_action_sequence[0].comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD
    assert comm_action_sequence[0].gather_dim == 1
    assert comm_action_sequence[0].logical_process_axis == 1

    # all-to-all(R, S0) -> [S0, R]
    assert comm_action_sequence[1].comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD
    assert comm_action_sequence[1].gather_dim == 1
    assert comm_action_sequence[1].shard_dim == 0
    assert comm_action_sequence[1].logical_process_axis == 0

    # shard(S0) -> [S01]
    assert comm_action_sequence[2].comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD
    assert comm_action_sequence[2].shard_dim == 0
    assert comm_action_sequence[2].logical_process_axis == 1

    assert shape_consistency_manager.cached_spec_pairs_transform_path[('[R, S01, R]',
                                                                       '[S01, R, R]')][0] == transform_path
    assert shape_consistency_manager.cached_spec_pairs_transform_path[('[R, S01, R]',
                                                                       '[S01, R, R]')][1] == comm_action_sequence


if __name__ == '__main__':
    test_one_step_transform()
    test_shape_consistency()
