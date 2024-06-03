import pytest
import torch
import torch.nn as nn

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import rerun_if_address_is_in_use, spawn
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy


class ConvFunctionModule(nn.Module):
    def __init__(self, in_channels=4, out_channels=64, kernel_size=3):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, input):
        return nn.functional.conv2d(input, self.conv_weight)


def _conv_module_mem_test(rank, world_size, port, bias):
    """This function is for conv memory test
    Test and print real memory cost and estimated, this test will not be executed except with the tag AUTO_PARALLEL

    Args:
    Args:
        rank: device rank
        bias: indicate whether conv module need bias
        world_size: number of devices
        port: port for initializing process group
    """
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = nn.Sequential(nn.Conv2d(4, 64, 3, padding=1, bias=bias)).cuda()
    input = torch.rand(4, 4, 64, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # index of target node in computation graph
    node_index = 1
    # total number of target node strategies
    strategy_number = 16
    mem_test_for_node_strategy(
        rank=rank,
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=[input],
        meta_arg_names=["input"],
    )


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_conv_meta_concrete_info_match(bias=False):
    spawn(_conv_module_mem_test, 4, bias=bias)


def _conv_function_mem_test(rank, world_size, port):
    """This function is for conv function memory test
    Test and print real memory cost and estimated, this test will not be executed except with the tag AUTO_PARALLEL

    Args:
        rank: device rank
        bias: indicate whether conv module need bias
        world_size: number of devices
        port: port for initializing process group
    """
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = ConvFunctionModule().cuda()
    input = torch.rand(4, 4, 64, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # index of target node in computation graph
    node_index = 2
    # total number of target node strategies
    strategy_number = 16
    mem_test_for_node_strategy(
        rank=rank,
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=[input],
        meta_arg_names=["input"],
    )


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_conv_function_concrete_info_match():
    spawn(_conv_function_mem_test, 4)


if __name__ == "__main__":
    # test_conv_meta_concrete_info_match()
    test_conv_function_concrete_info_match()
