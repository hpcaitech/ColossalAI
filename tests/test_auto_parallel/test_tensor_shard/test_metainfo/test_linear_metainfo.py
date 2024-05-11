import pytest
import torch
import torch.nn as nn

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import rerun_if_address_is_in_use, spawn
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy


class MyModule(nn.Module):
    def __init__(self, in_features=64, out_features=128):
        super().__init__()
        self.fc_weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, input):
        return nn.functional.linear(input, self.fc_weight)


def _linear_module_mem_test(rank, world_size, port):
    """This function is for linear memory test
    Test and print real memory cost and estimated, this test will not be executed except with the tag AUTO_PARALLEL

    Args:
        rank: device rank
        bias: indicate whether linear module need bias
        world_size: number of devices
        port: port for initializing process group
    """
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = nn.Sequential(nn.Linear(64, 128, bias=False)).cuda()
    input = torch.rand(8, 8, 16, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # memory test
    mem_test_for_node_strategy(
        rank=rank,
        model=model,
        device_mesh=device_mesh,
        node_index=1,
        strategy_number=13,
        input_args=[input],
        meta_arg_names=["input"],
    )


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_linear_module_meta_concrete_info_match():
    spawn(_linear_module_mem_test, 4)


def _linear_function_mem_test(rank, world_size, port):
    """This function is for linear memory test
    Test and print real memory cost and estimated, this test will not be executed except with the tag AUTO_PARALLEL

    Args:
        rank: device rank
        bias: indicate whether linear module need bias
        world_size: number of devices
        port: port for initializing process group
    """
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = MyModule().cuda()
    input = torch.rand(8, 8, 16, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # memory test
    mem_test_for_node_strategy(
        rank=rank,
        model=model,
        device_mesh=device_mesh,
        node_index=2,
        strategy_number=24,
        input_args=[input],
        meta_arg_names=["input"],
    )


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_linear_function_meta_concrete_info_match():
    spawn(_linear_function_mem_test, 4)


if __name__ == "__main__":
    # test_linear_module_meta_concrete_info_match()
    test_linear_function_meta_concrete_info_match()
