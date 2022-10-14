import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import (ConvFunctionHandler, ConvModuleHandler)
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (OperationData, OperationDataType, StrategiesVector)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.tracer.meta_patch.patched_module import linear


def test_conv_module_handler():
    model = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1).to('meta'))
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %_0 : [#users=1] = call_module[target=0](args = (%input_1,), kwargs = {})
    #     return _0
    graph = tracer.trace(model, meta_args={"input": torch.rand(4, 4, 64, 64).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    conv_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    handler = ConvModuleHandler(node=conv_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4, 64, 64])

    assert mapping['other'].name == "weight"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([16, 4, 3, 3])
    assert mapping['other'].type == OperationDataType.PARAM
    assert mapping['other'].logical_shape == torch.Size([4, 16, 3, 3])

    assert mapping['bias'].name == "bias"
    assert mapping['bias'].data.is_meta
    assert mapping['bias'].data.shape == torch.Size([16])
    assert mapping['bias'].type == OperationDataType.PARAM
    assert mapping['bias'].logical_shape == torch.Size([16])

    assert mapping['output'].name == "_0"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 16, 64, 64])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1' in strategy_name_list
    assert 'S1S0 = S1R x RS0' in strategy_name_list

    # SR = SR x RR
    assert 'S0R = S0R x RR' in strategy_name_list
    assert 'S1R = S1R x RR' in strategy_name_list

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

    # RR = RR x RR
    assert 'RR = RR x RR' in strategy_name_list

    # S01R = S01R x RR
    assert 'S01R = S01R x RR' in strategy_name_list

    # RR = RS01 x S01R
    assert 'RR = RS01 x S01R' in strategy_name_list

    # RS01 = RR x RS01
    assert 'RS01 = RR x RS01' in strategy_name_list


class ConvModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, others, bias=None):
        x = nn.functional.conv2d(input, others, bias=bias, padding=1)
        return x


def test_conv_function_handler():
    model = ConvModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %others : torch.Tensor [#users=1] = placeholder[target=others]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %others), kwargs = {})
    #     return conv2d
    graph = tracer.trace(model,
                         meta_args={
                             "input": torch.rand(4, 4, 64, 64).to('meta'),
                             "others": torch.rand(16, 4, 3, 3).to('meta'),
                             "bias": torch.rand(16).to('meta')
                         })
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    conv_mod_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    handler = ConvFunctionHandler(node=conv_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "input_1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4, 64, 64])

    assert mapping['other'].name == "others"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([16, 4, 3, 3])
    assert mapping['other'].type == OperationDataType.ARG
    assert mapping['other'].logical_shape == torch.Size([4, 16, 3, 3])

    assert mapping['bias'].name == "bias"
    assert mapping['bias'].data.is_meta
    assert mapping['bias'].data.shape == torch.Size([16])
    assert mapping['bias'].type == OperationDataType.ARG
    assert mapping['bias'].logical_shape == torch.Size([16])

    assert mapping['output'].name == "conv2d"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 16, 64, 64])
    assert mapping['output'].type == OperationDataType.OUTPUT

    handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1' in strategy_name_list
    assert 'S1S0 = S1R x RS0' in strategy_name_list

    # SR = SR x RR
    assert 'S0R = S0R x RR' in strategy_name_list
    assert 'S1R = S1R x RR' in strategy_name_list

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

    # RR = RR x RR
    assert 'RR = RR x RR' in strategy_name_list

    # S01R = S01R x RR
    assert 'S01R = S01R x RR' in strategy_name_list

    # RR = RS01 x S01R
    assert 'RR = RS01 x S01R' in strategy_name_list

    # RS01 = RR x RS01
    assert 'RS01 = RR x RS01' in strategy_name_list


if __name__ == '__main__':
    test_conv_module_handler()
    test_conv_function_handler()
