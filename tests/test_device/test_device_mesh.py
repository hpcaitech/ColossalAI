import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import rerun_if_address_is_in_use, spawn


def test_device_mesh():
    physical_mesh_id = torch.arange(0, 16)
    mesh_shape = (4, 4)
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    #  [8, 9, 10,11],
    #  [12,13,14,15]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    assert device_mesh.global_rank_to_local_rank(5) == [1, 1]
    assert device_mesh.global_rank_to_local_rank(11) == [2, 3]
    assert device_mesh.get_ranks_in_process_group(axis=1, global_rank=2) == [0, 1, 2, 3]


def check_1d_device_mesh():
    # check for 1D device mesh
    process_group = dist.GroupMember.WORLD
    device_mesh = DeviceMesh.from_process_group(process_group)

    # checks
    assert device_mesh.shape == [4]
    assert len(device_mesh.get_process_group_for_all_axes().keys()) == 1, "Expected 1 axis for the process group dict"
    assert device_mesh.get_process_group(axis=0) == process_group, "Expected world process group"
    assert device_mesh.is_initialized
    assert device_mesh.num_devices == 4
    assert device_mesh.is_initialized
    assert device_mesh.logical_mesh_id is None
    assert device_mesh._is_init_from_process_group


def check_2d_device_mesh():
    # create process group for 2D device mesh
    first_row_ranks = [0, 1]
    second_row_ranks = [2, 3]
    first_col_ranks = [0, 2]
    second_col_ranks = [1, 3]

    first_row_pg = dist.new_group(first_row_ranks, backend="nccl")
    second_row_pg = dist.new_group(second_row_ranks, backend="nccl")
    first_col_pg = dist.new_group(first_col_ranks, backend="nccl")
    second_col_pg = dist.new_group(second_col_ranks, backend="nccl")

    # check for
    current_rank = dist.get_rank()

    if current_rank in first_row_ranks:
        row_pg = first_row_pg
    else:
        row_pg = second_row_pg

    if current_rank in first_col_ranks:
        col_pg = first_col_pg
    else:
        col_pg = second_col_pg

    device_mesh = DeviceMesh.from_process_group([col_pg, row_pg])

    # checks
    assert device_mesh.shape == [2, 2]
    assert len(device_mesh.get_process_group_for_all_axes().keys()) == 2, "Expected 2 axes for the process group dict"
    assert device_mesh.get_process_group(axis=0) == col_pg, "Expected column process group"
    assert device_mesh.get_process_group(axis=1) == row_pg, "Expected row process group"
    assert device_mesh.num_devices == 4
    assert device_mesh.is_initialized
    assert device_mesh.logical_mesh_id is None
    assert device_mesh._is_init_from_process_group


def check_init_from_process_group(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_device_mesh_from_process_group():
    spawn(check_init_from_process_group, 4)


if __name__ == "__main__":
    test_device_mesh()
    test_device_mesh_from_process_group()
