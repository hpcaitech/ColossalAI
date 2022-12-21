from faulthandler import disable
from functools import partial
from xml.dom import WrongDocumentErr

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from typing_extensions import Self

from colossalai.auto_parallel.tensor_shard.node_handler import LinearFunctionHandler, LinearModuleHandler
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
from colossalai.testing.utils import parameterize
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


def check_linear_module_handler(rank, bias, input_shape, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = nn.Sequential(nn.Linear(16, 32, bias=bias)).cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    input = torch.rand(input_shape).cuda()
    # the index of linear node in computation graph
    node_index = 1
    # strategy number of linear node
    if input_shape == (1, 4, 4, 16):
        strategy_number = 19
    else:
        strategy_number = 24
    # construct input args
    input_args = [input]
    # construct meta arg names
    meta_arg_names = ['input']
    numerical_test_for_node_strategy(model=model,
                                     device_mesh=device_mesh,
                                     node_index=node_index,
                                     strategy_number=strategy_number,
                                     input_args=input_args,
                                     meta_arg_names=meta_arg_names)

    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={"input": torch.rand(input_shape).to('meta')})
    gm = ColoGraphModule(model, graph)

    linear_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = LinearModuleHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.shape == torch.Size(input_shape)
    assert mapping['input'].type == OperationDataType.ARG
    input_logical_shape = mapping['input'].data.view(-1, 16).shape
    assert mapping['input'].logical_shape == input_logical_shape

    assert mapping['other'].name == "weight"
    assert mapping['other'].data.shape == torch.Size([32, 16])
    assert mapping['other'].type == OperationDataType.PARAM
    assert mapping['other'].logical_shape == torch.Size([16, 32])

    if bias:
        assert mapping['bias'].name == "bias"
        assert mapping['bias'].data.shape == torch.Size([32])
        assert mapping['bias'].type == OperationDataType.PARAM
        assert mapping['bias'].logical_shape == torch.Size([32])

    assert mapping['output'].name == "_0"
    output_shape = input_shape[:-1] + (32,)
    assert mapping['output'].data.shape == torch.Size(output_shape)
    assert mapping['output'].type == OperationDataType.OUTPUT
    output_logical_shape = mapping['output'].data.view(-1, 32).shape
    assert mapping['output'].logical_shape == torch.Size(output_logical_shape)

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # First dimension cannot be shard if input shape is (1, 4, 4, 16)
    if input_shape != (1, 4, 4, 16):
        assert 'S1S0 = S1R x RS0_0' in strategy_name_list
        assert 'S0S1 = S0R x RS1_0' in strategy_name_list
        assert 'S1R = S1S0 x S0R_0' in strategy_name_list
        assert 'S0R = S0S1 x S1R_0' in strategy_name_list
        assert 'S01R = S01R x RR_0' in strategy_name_list

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1_1' in strategy_name_list
    assert 'S0S1 = S0R x RS1_2' in strategy_name_list
    assert 'S1S0 = S1R x RS0_1' in strategy_name_list
    assert 'S1S0 = S1R x RS0_2' in strategy_name_list

    # SR = SS x SR
    assert 'S0R = S0S1 x S1R_1' in strategy_name_list
    assert 'S0R = S0S1 x S1R_2' in strategy_name_list
    assert 'S1R = S1S0 x S0R_1' in strategy_name_list
    assert 'S1R = S1S0 x S0R_2' in strategy_name_list

    # RS = RS x SS
    assert 'RS0 = RS1 x S1S0' in strategy_name_list
    assert 'RS1 = RS0 x S0S1' in strategy_name_list

    # RR = RS x SR
    assert 'RR = RS0 x S0R' in strategy_name_list
    assert 'RR = RS1 x S1R' in strategy_name_list

    # RS= RR x RS
    assert 'RS0 = RR x RS0' in strategy_name_list
    assert 'RS1 = RR x RS1' in strategy_name_list

    # S01R = S01R x RR
    assert 'S01R = S01R x RR_1' in strategy_name_list
    assert 'S01R = S01R x RR_2' in strategy_name_list

    # RR = RS01 x S01R
    assert 'RR = RS01 x S01R' in strategy_name_list

    # RS01 = RR x RS01
    assert 'RS01 = RR x RS01' in strategy_name_list

    # RR = RR x RR
    assert 'RR = RR x RR' in strategy_name_list

    for strategy in strategies_vector:
        strategy: ShardingStrategy
        input_sharding_spec = strategy.get_sharding_spec_by_name('input_1')
        weight_sharding_spec = strategy.get_sharding_spec_by_name('weight')
        output_sharding_spec = strategy.get_sharding_spec_by_name('_0')

        if bias:
            bias_sharding_spec = strategy.get_sharding_spec_by_name('bias')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[:-1] == output_sharding_spec.sharding_sequence[:-1]
        assert weight_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert weight_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[-1]

        if bias:
            assert bias_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


class LinearModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, others, bias=None):
        x = nn.functional.linear(input, others, bias=bias)
        return x


def check_linear_function_handler(rank, bias, input_shape, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = LinearModel().cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    input = torch.rand(input_shape).cuda()
    other = torch.rand(32, 16).cuda()
    # the index of linear node in computation graph
    node_index = 2
    # strategy number of linear node
    if input_shape == (1, 4, 4, 16):
        strategy_number = 19
    else:
        strategy_number = 24
    # construct input args
    input_args = [input, other]
    # construct meta arg names
    meta_arg_names = ['input', 'others']
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
                             'others': torch.rand(32, 16).to('meta')
                         })
    gm = ColoGraphModule(model, graph)
    if bias:
        linear_func_node = list(graph.nodes)[3]
    else:
        linear_func_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(linear_func_node)

    # build handler
    handler = LinearFunctionHandler(node=linear_func_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.shape == torch.Size(input_shape)
    assert mapping['input'].type == OperationDataType.ARG
    input_logical_shape = mapping['input'].data.view(-1, 16).shape
    assert mapping['input'].logical_shape == torch.Size(input_logical_shape)

    assert mapping['other'].name == "others"
    assert mapping['other'].data.shape == torch.Size([32, 16])
    assert mapping['other'].type == OperationDataType.ARG
    assert mapping['other'].logical_shape == torch.Size([16, 32])

    if bias:
        assert mapping['bias'].name == "bias"
        assert mapping['bias'].data.shape == torch.Size([32])
        assert mapping['bias'].type == OperationDataType.ARG
        assert mapping['other'].logical_shape == torch.Size([16, 32])

    assert mapping['output'].name == "linear"
    output_shape = input_shape[:-1] + (32,)
    assert mapping['output'].data.shape == torch.Size(output_shape)
    assert mapping['output'].type == OperationDataType.OUTPUT
    output_logical_shape = mapping['output'].data.view(-1, 32).shape
    assert mapping['output'].logical_shape == torch.Size(output_logical_shape)

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # First dimension cannot be shard if input shape is (1, 4, 4, 16)
    if input_shape != (1, 4, 4, 16):
        assert 'S1S0 = S1R x RS0_0' in strategy_name_list
        assert 'S0S1 = S0R x RS1_0' in strategy_name_list
        assert 'S1R = S1S0 x S0R_0' in strategy_name_list
        assert 'S0R = S0S1 x S1R_0' in strategy_name_list
        assert 'S01R = S01R x RR_0' in strategy_name_list

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1_1' in strategy_name_list
    assert 'S0S1 = S0R x RS1_2' in strategy_name_list
    assert 'S1S0 = S1R x RS0_1' in strategy_name_list
    assert 'S1S0 = S1R x RS0_2' in strategy_name_list

    # SR = SS x SR
    assert 'S0R = S0S1 x S1R_1' in strategy_name_list
    assert 'S0R = S0S1 x S1R_2' in strategy_name_list
    assert 'S1R = S1S0 x S0R_1' in strategy_name_list
    assert 'S1R = S1S0 x S0R_2' in strategy_name_list

    # RS = RS x SS
    assert 'RS0 = RS1 x S1S0' in strategy_name_list
    assert 'RS1 = RS0 x S0S1' in strategy_name_list

    # RR = RS x SR
    assert 'RR = RS0 x S0R' in strategy_name_list
    assert 'RR = RS1 x S1R' in strategy_name_list

    # RS= RR x RS
    assert 'RS0 = RR x RS0' in strategy_name_list
    assert 'RS1 = RR x RS1' in strategy_name_list

    # S01R = S01R x RR
    assert 'S01R = S01R x RR_1' in strategy_name_list
    assert 'S01R = S01R x RR_2' in strategy_name_list

    # RR = RS01 x S01R
    assert 'RR = RS01 x S01R' in strategy_name_list

    # RS01 = RR x RS01
    assert 'RS01 = RR x RS01' in strategy_name_list

    # RR = RR x RR
    assert 'RR = RR x RR' in strategy_name_list

    for strategy in strategies_vector:
        strategy: ShardingStrategy
        input_sharding_spec = strategy.get_sharding_spec_by_name('input_1')
        weight_sharding_spec = strategy.get_sharding_spec_by_name('others')
        output_sharding_spec = strategy.get_sharding_spec_by_name('linear')

        if bias:
            bias_sharding_spec = strategy.get_sharding_spec_by_name('bias')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[:-1] == output_sharding_spec.sharding_sequence[:-1]
        assert weight_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert weight_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[-1]

        if bias:
            assert bias_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


@run_on_environment_flag(name='AUTO_PARALLEL')
@parameterize('input_shape', [(1, 4, 4, 16), (4, 4, 4, 16)])
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_linear_handler(input_shape, bias=False):
    world_size = 4
    run_func_module = partial(check_linear_module_handler,
                              bias=bias,
                              input_shape=input_shape,
                              world_size=world_size,
                              port=free_port())
    mp.spawn(run_func_module, nprocs=world_size)
    run_func_function = partial(check_linear_function_handler,
                                bias=bias,
                                input_shape=input_shape,
                                world_size=world_size,
                                port=free_port())
    mp.spawn(run_func_function, nprocs=world_size)


if __name__ == '__main__':
    test_linear_handler()
