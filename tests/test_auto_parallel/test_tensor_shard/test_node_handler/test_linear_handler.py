import torch
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
from colossalai.testing.utils import parameterize


@parameterize('bias', [True, False])
def test_linear_module_handler(bias):
    model = nn.Sequential(nn.Linear(16, 32, bias=bias).to('meta'))

    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={"input": torch.rand(2, 2, 4, 16).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    print(graph)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
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
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([2, 2, 4, 16])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([16, 16])

    assert mapping['other'].name == "weight"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([32, 16])
    assert mapping['other'].type == OperationDataType.PARAM
    assert mapping['other'].logical_shape == torch.Size([16, 32])

    if bias:
        assert mapping['bias'].name == "bias"
        assert mapping['bias'].data.is_meta
        assert mapping['bias'].data.shape == torch.Size([32])
        assert mapping['bias'].type == OperationDataType.PARAM
        assert mapping['bias'].logical_shape == torch.Size([32])

    assert mapping['output'].name == "_0"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([2, 2, 4, 32])
    assert mapping['output'].type == OperationDataType.OUTPUT
    assert mapping['output'].logical_shape == torch.Size([16, 32])

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
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


@parameterize('bias', [True, False])
def test_linear_function_handler(bias):
    model = nn.Linear(16, 32, bias=bias).to('meta')
    tracer = ColoTracer()
    graph = tracer.trace(model, meta_args={"input": torch.rand(2, 2, 4, 16).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    print(graph)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

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
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([2, 2, 4, 16])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([16, 16])

    assert mapping['other'].name == "weight"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([32, 16])
    assert mapping['other'].type == OperationDataType.PARAM
    assert mapping['other'].logical_shape == torch.Size([16, 32])

    if bias:
        assert mapping['bias'].name == "bias"
        assert mapping['bias'].data.is_meta
        assert mapping['bias'].data.shape == torch.Size([32])
        assert mapping['bias'].type == OperationDataType.PARAM
        assert mapping['other'].logical_shape == torch.Size([16, 32])

    assert mapping['output'].name == "linear"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([2, 2, 4, 32])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
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
        input_sharding_spec = strategy.get_sharding_spec_by_name('input_1')
        weight_sharding_spec = strategy.get_sharding_spec_by_name('weight')
        output_sharding_spec = strategy.get_sharding_spec_by_name('linear')

        if bias:
            bias_sharding_spec = strategy.get_sharding_spec_by_name('bias')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[:-1] == output_sharding_spec.sharding_sequence[:-1]
        assert weight_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert weight_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[-1]

        if bias:
            assert bias_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


if __name__ == '__main__':
    test_linear_module_handler()
    test_linear_function_handler()
