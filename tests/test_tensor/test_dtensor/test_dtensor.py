import torch

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor import ShardingSpec, distribute_tensor, get_global_shape, redistribute, to_global
from colossalai.testing import rerun_if_address_is_in_use, spawn


class TestModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features, out_features)
        self.linear_2 = torch.nn.Linear(out_features, in_features)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x


def check_dtensor(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    test_model = TestModel(8, 8).to("cuda")
    original_tensor = torch.rand(4, 8).to("cuda")
    compare_output = test_model(original_tensor)

    device_mesh = DeviceMesh(torch.Tensor([0, 1, 2, 3]), (2, 2), init_process_group=True)
    target_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict={0: [0]})
    d_tensor = distribute_tensor(original_tensor, device_mesh, target_sharding_spec)

    assert get_global_shape(d_tensor) == original_tensor.shape
    assert d_tensor.dtype == original_tensor.dtype

    if rank in (0, 1):
        assert d_tensor.equal(original_tensor.narrow(0, 0, 2))
    elif rank in (2, 3):
        assert d_tensor.equal(original_tensor.narrow(0, 2, 2))
    else:
        raise ValueError(f"rank {rank} is not in the device mesh")
    assert to_global(d_tensor).equal(original_tensor)
    output = test_model(d_tensor)

    if rank in (0, 1):
        assert output.equal(compare_output.narrow(0, 0, 2))
    elif rank in (2, 3):
        assert output.equal(compare_output.narrow(0, 2, 2))
    else:
        raise ValueError(f"rank {rank} is not in the device mesh")

    new_sharding_spec = ShardingSpec(dim_size=original_tensor.dim(), dim_partition_dict={0: [0, 1]})
    d_tensor = redistribute(d_tensor, device_mesh, new_sharding_spec)

    if rank == 0:
        assert d_tensor.equal(original_tensor.narrow(0, 0, 1))
    elif rank == 1:
        assert d_tensor.equal(original_tensor.narrow(0, 1, 1))
    elif rank == 2:
        assert d_tensor.equal(original_tensor.narrow(0, 2, 1))
    elif rank == 3:
        assert d_tensor.equal(original_tensor.narrow(0, 3, 1))
    else:
        raise ValueError(f"rank {rank} is not in the device mesh")

    dtensor_from_local = distribute_tensor(original_tensor, device_mesh, new_sharding_spec)

    if rank == 0:
        assert dtensor_from_local.equal(original_tensor.narrow(0, 0, 1))
    elif rank == 1:
        assert dtensor_from_local.equal(original_tensor.narrow(0, 1, 1))
    elif rank == 2:
        assert dtensor_from_local.equal(original_tensor.narrow(0, 2, 1))
    elif rank == 3:
        assert dtensor_from_local.equal(original_tensor.narrow(0, 3, 1))
    else:
        raise ValueError(f"rank {rank} is not in the device mesh")


@rerun_if_address_is_in_use()
def test_dtensor():
    world_size = 4
    spawn(check_dtensor, world_size)


if __name__ == "__main__":
    test_dtensor()
