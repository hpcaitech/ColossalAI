import pytest
import torch
import torch.nn as nn

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import rerun_if_address_is_in_use, spawn
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy


class BinaryElementwiseOpModule(nn.Module):
    def __init__(self, token=torch.add, shape=64) -> None:
        super().__init__()
        self.token = token
        self.param = nn.Parameter(torch.rand(shape))

    def forward(self, input):
        return input + self.param


def _binary_elementwise_mem_test(rank, world_size, port):
    """This function is for binary elementwise ops memory test
    Test and print real memory cost and estimated, this test will not be executed except with the tag AUTO_PARALLEL

    Args:
        rank: device rank
        bias: indicate whether conv module need bias
        world_size: number of devices
        port: port for initializing process group
    """
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = BinaryElementwiseOpModule(token=torch.add, shape=1024).cuda()
    input = torch.rand(32, 1024).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # index of target node in computation graph
    node_index = 2
    # total number of target node strategies
    strategy_number = 9
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
def test_binary_elementwise_meta_concrete_info_match():
    spawn(_binary_elementwise_mem_test, 4)


if __name__ == "__main__":
    test_binary_elementwise_meta_concrete_info_match()
