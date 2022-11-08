from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy


def _batchnorm_module_mem_test(rank, world_size, port):
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
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = nn.Sequential(nn.BatchNorm2d(128)).cuda()
    input = torch.rand(4, 128, 64, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # index of conv node in computation graph
    node_index = 1
    # total number of conv strategies
    strategy_number = 4
    mem_test_for_node_strategy(rank=rank,
                               model=model,
                               device_mesh=device_mesh,
                               node_index=node_index,
                               strategy_number=strategy_number,
                               input_args=[input],
                               meta_arg_names=['input'])


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_batchnorm_meta_concrete_info_match():
    world_size = 4
    run_func_module = partial(_batchnorm_module_mem_test, world_size=world_size, port=free_port())
    mp.spawn(run_func_module, nprocs=world_size)


if __name__ == '__main__':
    test_batchnorm_meta_concrete_info_match()
