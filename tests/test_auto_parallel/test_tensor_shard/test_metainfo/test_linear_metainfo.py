from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import LinearModuleHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.testing.utils import parameterize, rerun_if_address_is_in_use
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_metainfo.utils import mem_test_for_node_strategy

if torch.__version__ >= '1.12.0':
    from colossalai.auto_parallel.meta_profiler import MetaInfo, meta_register


@pytest.mark.skipif(torch.__version__ < '1.12.0', reason='PyTorch version is too low')
@parameterize('bias', [True, False])
def test_linear_metainfo(bias):
    model = nn.Sequential(nn.Linear(16, 32, bias=bias).to('meta'))

    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={"input": torch.rand(2, 2, 4, 16).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    linear_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = LinearModuleHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # build strategy
    strategies_vector = handler.register_strategy(compute_resharding_cost=False)

    # assert module is registered
    assert meta_register.has(linear_mod_node.graph.owning_module.get_submodule(linear_mod_node.target).__class__)

    # check metainfo
    for strategy in strategies_vector:
        strategy: ShardingStrategy
        try:
            metainfo = MetaInfo(strategy,
                                linear_mod_node.graph.owning_module.get_submodule(linear_mod_node.target).__class__)

        except:
            raise RuntimeError(f"Failed to compute metainfo for {strategy}")


def _linear_mem_test(rank, bias, world_size, port):
    """This function is for linear memory test
    Test and print real memory cost and estimated, this test will not be executed
    in unit test.

    Args:
        bias (bool, optional): Indicate whether we need bias for Linear. Defaults to True.
    """
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = nn.Sequential(nn.Linear(64, 128, bias=bias)).cuda()
    input = torch.rand(8, 8, 16, 64).cuda()
    input.requires_grad = True
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # memory test
    mem_test_for_node_strategy(rank=rank,
                               model=model,
                               device_mesh=device_mesh,
                               node_index=1,
                               strategy_number=13,
                               input_args=[input],
                               meta_arg_names=["input"])


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_linear_meta_concrete_info_match(bias=False):
    world_size = 4
    run_func_module = partial(_linear_mem_test, bias=bias, world_size=world_size, port=free_port())
    mp.spawn(run_func_module, nprocs=world_size)


if __name__ == '__main__':
    # test_linear_metainfo()
    # _linear_mem_test(bias=True)
    test_linear_meta_concrete_info_match()
