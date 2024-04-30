import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor import ShardingSpec, distribute_tensor, is_distributed_tensor, to_global
from colossalai.tensor.padded_tensor import is_padded_tensor, to_padded_tensor, to_unpadded_tensor
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_padded_tensor(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    original_tensor = torch.rand(32, 64).to("cuda")

    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    target_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict={0: [0]})
    d_tensor = distribute_tensor(original_tensor, device_mesh, target_sharding_spec)

    padded_tensor = to_padded_tensor(d_tensor, current_length=64, padding_dim=0)
    assert padded_tensor.dist_layout == d_tensor.dist_layout

    tensor_copy = padded_tensor.clone()
    assert is_padded_tensor(tensor_copy)
    assert is_distributed_tensor(tensor_copy)

    tensor_detached = padded_tensor.detach()
    assert is_padded_tensor(tensor_detached)
    assert is_distributed_tensor(tensor_detached)

    unpadded_tensor = to_unpadded_tensor(padded_tensor)
    assert unpadded_tensor.shape == d_tensor.shape
    assert is_distributed_tensor(unpadded_tensor)

    global_tensor = to_global(unpadded_tensor)
    assert global_tensor.shape == original_tensor.shape


@rerun_if_address_is_in_use()
def test_padded_tensor():
    world_size = 4
    spawn(check_padded_tensor, world_size)


if __name__ == "__main__":
    test_padded_tensor()
