import pytest
import torch
import torch.distributed as dist

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.comm_spec import CollectiveCommPattern, CommSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_all_gather(process_groups_dict, rank):
    # tensor to comm
    if rank in (0, 2):
        sharded_tensor_to_comm = torch.ones(2, 2).cuda()
    else:
        sharded_tensor_to_comm = torch.zeros(2, 2).cuda()

    # tensor to check
    tensor_to_check = torch.cat((torch.ones(2, 2), torch.zeros(2, 2)), 1).cuda()

    # CommSpec:(comm_pattern:allgather, gather_dim:1, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.GATHER_FWD_SPLIT_BWD, process_groups_dict, gather_dim=1, logical_process_axis=1
    )
    sharded_tensor_to_comm = sharded_tensor_to_comm = comm_spec.covert_spec_to_action(sharded_tensor_to_comm)

    assert sharded_tensor_to_comm.equal(tensor_to_check)


def check_shard(process_groups_dict, rank):
    # tensor to comm
    sharded_tensor_to_comm_0 = torch.zeros(2, 2).cuda()
    sharded_tensor_to_comm_1 = torch.ones(2, 2).cuda()
    # tensor([[0., 0., 1., 1.],
    #         [0., 0., 1., 1.]])
    tensor_to_shard = torch.cat((sharded_tensor_to_comm_0, sharded_tensor_to_comm_1), 1)

    # CommSpec:(comm_pattern:shard, shard_dim:1, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.SPLIT_FWD_GATHER_BWD, process_groups_dict, shard_dim=1, logical_process_axis=1
    )
    tensor_to_shard = comm_spec.covert_spec_to_action(tensor_to_shard)

    if rank in (0, 2):
        assert tensor_to_shard.equal(sharded_tensor_to_comm_0)
    if rank in (1, 3):
        assert tensor_to_shard.equal(sharded_tensor_to_comm_1)


def check_all_to_all(process_groups_dict, rank):
    # tensor to comm
    if rank in (0, 1):
        sharded_tensor_0 = torch.zeros(2, 1)
        sharded_tensor_1 = torch.ones(2, 1)
        # tensor([[0., 1.],
        #         [0., 1.]])
        tensor_to_comm = torch.cat((sharded_tensor_0, sharded_tensor_1), 1).cuda()
    if rank in (2, 3):
        sharded_tensor_0 = torch.ones(2, 1) * 2
        sharded_tensor_1 = torch.ones(2, 1) * 3
        # tensor([[2., 3.],
        #         [2., 3.]])
        tensor_to_comm = torch.cat((sharded_tensor_0, sharded_tensor_1), 1).cuda()

    if rank in (0, 1):
        # tensor([[0.],
        #         [0.],
        #         [2.],
        #         [2.]])
        tensor_to_check = torch.tensor([[0], [0], [2], [2]], dtype=tensor_to_comm.dtype).cuda()
    if rank in (2, 3):
        # tensor([[1.],
        #         [1.],
        #         [3.],
        #         [3.]])
        tensor_to_check = torch.tensor([[1], [1], [3], [3]], dtype=tensor_to_comm.dtype).cuda()

    # CommSpec:(comm_pattern:shard, shard_dim:1, logical_process_axis:1)
    comm_spec = CommSpec(
        CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD,
        process_groups_dict,
        gather_dim=0,
        shard_dim=1,
        logical_process_axis=0,
    )
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_all_reduce_fwd(process_groups_dict, rank):
    # tensor to comm
    tensor_to_comm = torch.ones(2, 2).cuda() * rank

    # reduce through logical process axis 0
    # tensor to check
    if rank in (0, 2):
        # tensor([[2., 2.],
        #         [2., 2.]])
        tensor_to_check = torch.tensor([[2, 2], [2, 2]], dtype=tensor_to_comm.dtype).cuda()
    if rank in (1, 3):
        # tensor([[4., 4.],
        #         [4., 4.]])
        tensor_to_check = torch.tensor([[4, 4], [4, 4]], dtype=tensor_to_comm.dtype).cuda()

    comm_spec = CommSpec(CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD, process_groups_dict, logical_process_axis=0)
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_all_reduce_bwd(process_groups_dict, rank):
    # tensor to comm
    tensor_to_comm = torch.ones(2, 2).cuda() * rank

    tensor_to_check = torch.ones(2, 2).cuda() * rank

    comm_spec = CommSpec(CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD, process_groups_dict, logical_process_axis=0)
    tensor_to_comm = comm_spec.covert_spec_to_action(tensor_to_comm)

    assert tensor_to_comm.equal(tensor_to_check)


def check_comm(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    physical_mesh_id = torch.arange(0, 4)
    assert rank == dist.get_rank()

    mesh_shape = (2, 2)
    # [[0, 1,
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    process_group_dict = device_mesh._process_group_dict[rank]

    # test all gather
    check_all_gather(process_group_dict, rank)

    # test shard
    check_shard(process_group_dict, rank)

    # test all to all
    check_all_to_all(process_group_dict, rank)

    # test all reduce
    check_all_reduce_fwd(process_group_dict, rank)
    check_all_reduce_bwd(process_group_dict, rank)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_comm_spec():
    world_size = 4
    spawn(check_comm, world_size)


if __name__ == "__main__":
    test_comm_spec()
