import pytest
import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_apply(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1,
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    entire_shape = torch.Size((4, 2))
    shape_consistency_manager = ShapeConsistencyManager()
    dim_partition_source = {0: [0]}
    dim_partition_target = {1: [0]}

    # DistSpec:
    #     shard_sequence: S0,R
    #     device_mesh_shape: (2, 2)
    sharding_spec_source = ShardingSpec(device_mesh, entire_shape, dim_partition_source)

    # DistSpec:
    #     shard_sequence: R,S0
    #     device_mesh_shape: (2, 2)
    sharding_spec_target = ShardingSpec(device_mesh, entire_shape, dim_partition_target)

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

    tensor_to_comm.sharding_spec = sharding_spec_source
    tensor_to_comm = shape_consistency_manager.apply(tensor_to_comm, sharding_spec_target)
    assert tensor_to_comm.equal(tensor_to_check)
    assert str(tensor_to_comm.sharding_spec.sharding_sequence) == str(sharding_spec_target.sharding_sequence)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_apply():
    world_size = 4
    spawn(check_apply, world_size)


if __name__ == "__main__":
    test_apply()
