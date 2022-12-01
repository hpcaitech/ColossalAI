from faulthandler import disable
from functools import partial
from xml.dom import WrongDocumentErr

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from typing_extensions import Self

from colossalai.auto_parallel.tensor_shard.node_handler import ADDMMFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


class AddmmModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, m1, m2):
        x = torch.addmm(input, m1, m2)
        return x


def check_linear_function_handler(rank, input_shape, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = AddmmModel().cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    input = torch.rand(input_shape).cuda()
    m1 = torch.rand(4, 8).cuda()
    m2 = torch.rand(8, 16).cuda()
    # the index of addmm node in computation graph
    node_index = 3
    # strategy number of linear node
    strategy_number = 10
    # construct input args
    input_args = [input, m1, m2]
    # construct meta arg names
    meta_arg_names = ['input', 'm1', 'm2']
    numerical_test_for_node_strategy(model=model,
                                     device_mesh=device_mesh,
                                     node_index=node_index,
                                     strategy_number=strategy_number,
                                     input_args=input_args,
                                     meta_arg_names=meta_arg_names)

    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             "input": torch.rand(input_shape).to('meta'),
                             'm1': torch.rand(4, 8).to('meta'),
                             'm2': torch.rand(8, 16).to('meta'),
                         })
    gm = ColoGraphModule(model, graph)
    # [input_1, m1, m2, addmm, output]
    node_list = list(graph.nodes)
    addmm_node = node_list[3]
    strategies_vector = StrategiesVector(addmm_node)

    # build handler
    handler = ADDMMFunctionHandler(node=addmm_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    assert mapping['input'].name == "m1"
    assert mapping['input'].data.shape == torch.Size([4, 8])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 8])

    assert mapping['other'].name == "m2"
    assert mapping['other'].data.shape == torch.Size([8, 16])
    assert mapping['other'].type == OperationDataType.ARG
    assert mapping['other'].logical_shape == torch.Size([8, 16])

    assert mapping['bias'].name == "input_1"
    assert mapping['bias'].data.shape == torch.Size(input_shape)
    assert mapping['bias'].type == OperationDataType.ARG
    assert mapping['bias'].logical_shape == torch.Size([4, 16])

    assert mapping['output'].name == "addmm"
    assert mapping['output'].data.shape == torch.Size([4, 16])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # one strategy will be converted to different physical sharding spec
    assert len(strategy_name_list) > 8

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1' in strategy_name_list
    assert 'S1S0 = S1R x RS0' in strategy_name_list

    # SR = SS x SR
    assert 'S0R = S0S1 x S1R' in strategy_name_list
    assert 'S1R = S1S0 x S0R' in strategy_name_list

    # RS = RS x SS
    assert 'RS0 = RS1 x S1S0' in strategy_name_list
    assert 'RS1 = RS0 x S0S1' in strategy_name_list

    # RR = RS x SR
    assert 'RR = RS0 x S0R' in strategy_name_list
    assert 'RR = RS1 x S1R' in strategy_name_list

    # RS= RR x RS
    assert 'RS0 = RR x RS0' in strategy_name_list
    assert 'RS1 = RR x RS1' in strategy_name_list

    for strategy in strategies_vector:
        strategy: ShardingStrategy
        input_sharding_spec = strategy.get_sharding_spec_by_name('m1')
        weight_sharding_spec = strategy.get_sharding_spec_by_name('m2')
        output_sharding_spec = strategy.get_sharding_spec_by_name('addmm')
        bias_sharding_spec = strategy.get_sharding_spec_by_name('input_1')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[:-1] == output_sharding_spec.sharding_sequence[:-1]
        assert weight_sharding_spec.sharding_sequence[0] == input_sharding_spec.sharding_sequence[1]
        assert weight_sharding_spec.sharding_sequence[1] == output_sharding_spec.sharding_sequence[1]
        assert bias_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


@parameterize('input_shape', [(16,), (4, 16)])
@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_addmm_handler(input_shape):
    world_size = 4
    run_func_function = partial(check_linear_function_handler,
                                input_shape=input_shape,
                                world_size=world_size,
                                port=free_port())
    mp.spawn(run_func_function, nprocs=world_size)


if __name__ == '__main__':
    test_addmm_handler()
