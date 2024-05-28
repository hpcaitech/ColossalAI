import pytest
import torch
import torch.distributed as dist

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.shape_consistency import CollectiveCommPattern, CommSpec
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.tensor.utils import mix_gather_simulator
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_mix_gather_S0S1(device_mesh, rank):
    tensor_to_check = torch.arange(64).reshape((8, 8)).cuda()
    (f, b) = (0, 1)
    f_target_pair = (f, [0])
    b_target_pair = (b, [1])
    gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
    tensor_slice = [4, 2]  # (4, 2)
    rank_slice = 4
    f_start = (rank // rank_slice) * tensor_slice[0]
    b_start = (rank % rank_slice) * tensor_slice[1]
    tensor_to_comm = (
        tensor_to_check[f_start : f_start + tensor_slice[0], b_start : b_start + tensor_slice[1]].contiguous().cuda()
    )

    dim_partition_dict = {0: [0], 1: [1]}

    # DistSpec:
    #     shard_sequence: S0,S1
    #     device_mesh_shape: (2, 4)
    source_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    comm_spec = CommSpec(
        CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD,
        sharding_spec=source_spec,
        gather_dim=gather_dim,
        logical_process_axis=logical_process_axes,
        forward_only=True,
        mix_gather=True,
    )
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_two_all_gather_S0S1(device_mesh, rank):
    tensor_width = 8
    tensor_to_check = torch.arange(int(tensor_width * tensor_width)).reshape((tensor_width, tensor_width)).cuda()

    dim_partition_dict = {0: [0], 1: [1]}

    tensor_slice = [tensor_width // 2, tensor_width // 4]  # (4, 2)
    rank_slice = 4
    f_start = (rank // rank_slice) * tensor_slice[0]
    b_start = (rank % rank_slice) * tensor_slice[1]
    tensor_to_comm = (
        tensor_to_check[f_start : f_start + tensor_slice[0], b_start : b_start + tensor_slice[1]].contiguous().cuda()
    )

    # DistSpec:
    #     shard_sequence: S0,S1
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:0, logical_process_axis:0)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=0, logical_process_axis=0
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    dim_partition_dict = {1: [1]}
    # DistSpec:
    #     shard_sequence: R,S1
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=1, logical_process_axis=1
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_mix_gather_S1S0(device_mesh, rank):
    tensor_to_check = torch.arange(64).reshape((8, 8)).cuda()
    (f, b) = (0, 1)
    f_target_pair = (f, [1])
    b_target_pair = (b, [0])
    gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
    tensor_slice = [2, 4]
    rank_slice = 4
    f_start = (rank % rank_slice) * tensor_slice[0]
    b_start = (rank // rank_slice) * tensor_slice[1]
    tensor_to_comm = (
        tensor_to_check[f_start : f_start + tensor_slice[0], b_start : b_start + tensor_slice[1]].contiguous().cuda()
    )

    dim_partition_dict = {0: [1], 1: [0]}

    # DistSpec:
    #     shard_sequence: S1,S0
    #     device_mesh_shape: (2, 4)
    source_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    comm_spec = CommSpec(
        CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD,
        sharding_spec=source_spec,
        gather_dim=gather_dim,
        logical_process_axis=logical_process_axes,
        forward_only=True,
        mix_gather=True,
    )
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_two_all_gather_S1S0(device_mesh, rank):
    tensor_width = 8
    tensor_to_check = torch.arange(int(tensor_width * tensor_width)).reshape((tensor_width, tensor_width)).cuda()

    tensor_slice = [tensor_width // 4, tensor_width // 2]  # (4, 2)
    rank_slice = 4
    f_start = (rank % rank_slice) * tensor_slice[0]
    b_start = (rank // rank_slice) * tensor_slice[1]
    tensor_to_comm = (
        tensor_to_check[f_start : f_start + tensor_slice[0], b_start : b_start + tensor_slice[1]].contiguous().cuda()
    )

    dim_partition_dict = {0: [1], 1: [0]}

    # DistSpec:
    #     shard_sequence: S1,S0
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:0, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=0, logical_process_axis=1
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    dim_partition_dict = {1: [0]}
    # DistSpec:
    #     shard_sequence: R,S0
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:0)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=1, logical_process_axis=0
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_mix_gather_S01R(device_mesh, rank):
    tensor_to_check = torch.arange(64).reshape((8, 8)).cuda()
    (f, b) = (0, 1)
    f_target_pair = (f, [0, 1])
    b_target_pair = (b, [])
    gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
    tensor_to_comm = tensor_to_check[rank : rank + 1, :].contiguous().cuda()

    dim_partition_dict = {0: [0, 1]}
    # DistSpec:
    #     shard_sequence: S01,R
    #     device_mesh_shape: (2, 4)
    source_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    comm_spec = CommSpec(
        CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD,
        sharding_spec=source_spec,
        gather_dim=gather_dim,
        logical_process_axis=logical_process_axes,
        forward_only=True,
        mix_gather=True,
    )
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_two_all_gather_S01R(device_mesh, rank):
    tensor_width = 8
    tensor_to_check = torch.arange(int(tensor_width * tensor_width)).reshape((tensor_width, tensor_width)).cuda()

    rank_stride = tensor_width // 8
    tensor_to_comm = tensor_to_check[rank : rank + rank_stride, :].contiguous().cuda()

    dim_partition_dict = {0: [0, 1]}

    # DistSpec:
    #     shard_sequence: S01, R
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:0, logical_process_axis:0)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=0, logical_process_axis=1
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    dim_partition_dict = {0: [0]}

    # DistSpec:
    #     shard_sequence: S1, R
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:0, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=0, logical_process_axis=0
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_mix_gather_RS01(device_mesh, rank):
    tensor_to_check = torch.arange(64).reshape((8, 8)).cuda()

    (f, b) = (0, 1)
    f_target_pair = (f, [])
    b_target_pair = (b, [0, 1])
    gather_dim, logical_process_axes = mix_gather_simulator(f_target_pair, b_target_pair)
    tensor_to_comm = tensor_to_check[:, rank : rank + 1].contiguous().cuda()

    dim_partition_dict = {1: [0, 1]}
    # DistSpec:
    #     shard_sequence: R, S01
    #     device_mesh_shape: (2, 4)
    source_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    comm_spec = CommSpec(
        CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD,
        sharding_spec=source_spec,
        gather_dim=gather_dim,
        logical_process_axis=logical_process_axes,
        forward_only=True,
        mix_gather=True,
    )
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_two_all_gather_RS01(device_mesh, rank):
    tensor_width = 8
    tensor_to_check = torch.arange(int(tensor_width * tensor_width)).reshape((tensor_width, tensor_width)).cuda()

    rank_stride = tensor_width // 8
    tensor_to_comm = tensor_to_check[:, rank : rank + rank_stride].contiguous().cuda()

    dim_partition_dict = {1: [0, 1]}

    # DistSpec:
    #     shard_sequence: R, S01
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:0)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=1, logical_process_axis=1
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    dim_partition_dict = {1: [0]}

    # DistSpec:
    #     shard_sequence: R, S1
    #     device_mesh_shape: (2, 4)
    sharding_spec = ShardingSpec(device_mesh, tensor_to_check.shape, dim_partition_dict=dim_partition_dict)

    # CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, sharding_spec, gather_dim=1, logical_process_axis=0
    )

    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_comm(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    physical_mesh_id = torch.arange(0, 8)
    assert rank == dist.get_rank()

    mesh_shape = (2, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True, need_flatten=True)

    check_mix_gather_S0S1(device_mesh, rank)

    check_two_all_gather_S0S1(device_mesh, rank)

    check_mix_gather_S1S0(device_mesh, rank)

    check_two_all_gather_S1S0(device_mesh, rank)

    check_mix_gather_S01R(device_mesh, rank)

    check_two_all_gather_S01R(device_mesh, rank)

    check_mix_gather_RS01(device_mesh, rank)

    check_two_all_gather_RS01(device_mesh, rank)


@pytest.mark.skip(reason="Skip because the check functions assume 8 GPUS but CI only have 4 GPUs")
@rerun_if_address_is_in_use()
def test_mix_gather():
    world_size = 8
    spawn(check_comm, world_size)


if __name__ == "__main__":
    test_mix_gather()
