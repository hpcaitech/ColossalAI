import math

import pytest
import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.comm_spec import CollectiveCommPattern
from colossalai.tensor.d_tensor.layout import Layout
from colossalai.tensor.d_tensor.layout_converter import LayoutConverter
from colossalai.tensor.d_tensor.sharding_spec import ShardingSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn

global_shape = torch.Size((64, 32, 16))
layout_converter = LayoutConverter()
physical_mesh_id = torch.arange(0, 4)
mesh_shape = (2, 2)


def check_one_step_transform(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # [[0, 1],
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    dim_partition_dict = {0: [0], 1: [1]}
    # DistSpec:
    #     shard_sequence: S0,S1,R
    #     device_mesh_shape: (2, 2)
    sharding_spec = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_dict)
    layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec, global_shape=global_shape)

    rst_dict = layout_converter.all_gather_transform_layouts(layout)

    assert "[R, S1, R]" in [
        str(all_gather_layout.sharding_spec.sharding_sequence) for all_gather_layout in rst_dict.keys()
    ]
    assert "[S0, R, R]" in [
        str(all_gather_layout.sharding_spec.sharding_sequence) for all_gather_layout in rst_dict.keys()
    ]

    dim_partition_dict_all2all = {0: [0], 1: [1]}
    # DistSpec:
    #     shard_sequence: S0,S1,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_all2all = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_dict_all2all)
    layout_all2all = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_all2all, global_shape=global_shape)

    rst_dict_all2all = layout_converter.all_to_all_transform_layout(layout_all2all)

    assert "[S01, R, R]" in [
        str(all2all_layout.sharding_spec.sharding_sequence) for all2all_layout in rst_dict_all2all.keys()
    ]
    assert "[R, S1, S0]" in [
        str(all2all_layout.sharding_spec.sharding_sequence) for all2all_layout in rst_dict_all2all.keys()
    ]
    assert "[S0, R, S1]" in [
        str(all2all_layout.sharding_spec.sharding_sequence) for all2all_layout in rst_dict_all2all.keys()
    ]

    dim_partition_shard = {0: [0]}
    # DistSpec:
    #     shard_sequence: S0,R,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_shard = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_shard)
    shard_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_shard, global_shape=global_shape)

    rst_dict_shard = layout_converter.shard_transform_layout(shard_layout)

    assert "[S01, R, R]" in [
        str(shard_layout.sharding_spec.sharding_sequence) for shard_layout in rst_dict_shard.keys()
    ]
    assert "[S0, S1, R]" in [
        str(shard_layout.sharding_spec.sharding_sequence) for shard_layout in rst_dict_shard.keys()
    ]
    assert "[S0, R, S1]" in [
        str(shard_layout.sharding_spec.sharding_sequence) for shard_layout in rst_dict_shard.keys()
    ]


def check_layout_converting(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    dim_partition_source = {1: [0, 1]}
    dim_partition_target = {0: [0, 1]}
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # DistSpec:
    #     shard_sequence: R,S01,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_source = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_source)
    source_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_source, global_shape=global_shape)

    # DistSpec:
    #     shard_sequence: S01,R,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_target = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_target)
    target_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_target, global_shape=global_shape)

    transform_path, comm_action_sequence = layout_converter.layout_converting(source_layout, target_layout)

    # check transform path
    transform_path_str = "->".join([str(layout.sharding_spec.sharding_sequence) for layout in transform_path])
    assert transform_path_str == "[R, S01, R]->[R, S0, R]->[S0, R, R]->[S01, R, R]"

    # check comm action sequence
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

    # checkout chached_spec_pairs_transform_path
    src_shape = source_layout.get_sharded_shape_per_device()
    dst_shape = target_layout.get_sharded_shape_per_device()
    assert (
        layout_converter.cached_solution[(("[R, S01, R]", src_shape), ("[S01, R, R]", dst_shape))][0] == transform_path
    )
    assert (
        layout_converter.cached_solution[(("[R, S01, R]", src_shape), ("[S01, R, R]", dst_shape))][1]
        == comm_action_sequence
    )

    comm_cost = layout_converter.get_total_comm_cost(source_layout, target_layout)

    assert comm_cost["forward"] == comm_cost["backward"]
    assert math.floor(comm_cost["total"]) == math.floor(comm_cost["forward"] + comm_cost["backward"])


def check_layout_converting_apply(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    dim_partition_source = {1: [0, 1]}
    dim_partition_target = {0: [0, 1]}
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # DistSpec:
    #     shard_sequence: R,S01,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_source = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_source)
    source_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_source, global_shape=global_shape)

    # DistSpec:
    #     shard_sequence: S01,R,R
    #     device_mesh_shape: (4, 4)
    sharding_spec_target = ShardingSpec(dim_size=3, dim_partition_dict=dim_partition_target)
    target_layout = Layout(device_mesh=device_mesh, sharding_spec=sharding_spec_target, global_shape=global_shape)

    original_tensor = torch.rand(global_shape).cuda()

    # tensor_to_apply: [R, S01, R]
    tensor_to_apply = original_tensor.narrow(1, rank * 8, 8)

    # tensor_to_check: [S01, R, R]
    tensor_to_check = original_tensor.narrow(0, rank * 16, 16)

    converted_tensor = layout_converter.apply(tensor_to_apply, source_layout, target_layout)
    assert converted_tensor.equal(tensor_to_check)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_layout_converter():
    world_size = 4
    spawn(check_one_step_transform, world_size)
    spawn(check_layout_converting, world_size)
    spawn(check_layout_converting_apply, world_size)


if __name__ == "__main__":
    test_layout_converter()
