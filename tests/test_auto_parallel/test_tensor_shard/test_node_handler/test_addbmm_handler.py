import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import AddBMMFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing import parameterize


class AddBMMTensorMethodModule(nn.Module):

    def forward(self, bias, x1, x2):
        return bias.addbmm(x1, x2)


class AddBMMTorchFunctionModule(nn.Module):

    def forward(self, bias, x1, x2):
        return torch.addbmm(bias, x1, x2)


@parameterize('module', [AddBMMTorchFunctionModule, AddBMMTensorMethodModule])
@parameterize('bias_shape', [[8], [1, 8], [8, 8]])
def test_2d_device_mesh(module, bias_shape):

    model = module()
    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             'bias': torch.rand(*bias_shape).to('meta'),
                             "x1": torch.rand(4, 8, 16).to('meta'),
                             'x2': torch.rand(4, 16, 8).to('meta')
                         })
    print(graph)
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    linear_mod_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = AddBMMFunctionHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "x1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 8, 16])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 8, 16])

    assert mapping['other'].name == "x2"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([4, 16, 8])
    assert mapping['other'].type == OperationDataType.ARG
    assert mapping['other'].logical_shape == torch.Size([4, 16, 8])

    assert mapping['bias'].name == "bias"
    assert mapping['bias'].data.is_meta
    assert mapping['bias'].data.shape == torch.Size(bias_shape)
    assert mapping['bias'].type == OperationDataType.ARG
    assert mapping['bias'].logical_shape == torch.Size([8, 8])

    assert mapping['output'].name == "addbmm"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([8, 8])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # one batch dim
    assert 'Sb0 = Sb0 x Sb0' not in strategy_name_list

    # two batch dim
    assert 'Sb01 = Sb01 x Sb01' in strategy_name_list

    # SbSi = SbSi x Sb
    assert 'Sb0Si1 = Sb0Si1 x Sb0' in strategy_name_list
    assert 'Sb1Si0 = Sb1Si0 x Sb1' in strategy_name_list

    # SbSj = SbR x SbSj
    assert 'Sb0Sj1 = Sb0R x Sb0Sj1' in strategy_name_list
    assert 'Sb1Sj0 = Sb1R x Sb1Sj0' in strategy_name_list

    # SbR = SbSk x SbSk
    assert 'Sb0R = Sb0Sk1 x Sb0Sk1' in strategy_name_list
    assert 'Sb1R = Sb1Sk0 x Sb1Sk0' in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name('x1')
        other_sharding_spec = strategy.get_sharding_spec_by_name('x2')
        bias_sharding_spec = strategy.get_sharding_spec_by_name('bias')
        output_sharding_spec = strategy.get_sharding_spec_by_name('addbmm')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[1] == output_sharding_spec.sharding_sequence[0]
        assert other_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert other_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]
        assert bias_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


@parameterize('module', [AddBMMTorchFunctionModule, AddBMMTensorMethodModule])
@parameterize('bias_shape', [[8], [1, 8], [8, 8]])
def test_1d_device_mesh(module, bias_shape):
    model = module()
    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             'bias': torch.rand(*bias_shape).to('meta'),
                             "x1": torch.rand(4, 8, 16).to('meta'),
                             'x2': torch.rand(4, 16, 8).to('meta')
                         })
    print(graph)
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (1, 4)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    linear_mod_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = AddBMMFunctionHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "x1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 8, 16])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 8, 16])

    assert mapping['other'].name == "x2"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([4, 16, 8])
    assert mapping['other'].type == OperationDataType.ARG
    assert mapping['other'].logical_shape == torch.Size([4, 16, 8])

    assert mapping['bias'].name == "bias"
    assert mapping['bias'].data.is_meta
    assert mapping['bias'].data.shape == torch.Size(bias_shape)
    assert mapping['bias'].type == OperationDataType.ARG
    assert mapping['bias'].logical_shape == torch.Size([8, 8])

    assert mapping['output'].name == "addbmm"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([8, 8])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
    assert len(strategy_name_list) == 1
    # one batch dim
    assert 'Sb0 = Sb0 x Sb0' in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name('x1')
        other_sharding_spec = strategy.get_sharding_spec_by_name('x2')
        bias_sharding_spec = strategy.get_sharding_spec_by_name('bias')
        output_sharding_spec = strategy.get_sharding_spec_by_name('addbmm')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[1] == output_sharding_spec.sharding_sequence[0]
        assert other_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert other_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]
        assert bias_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


if __name__ == '__main__':
    test_1d_device_mesh()
    # test_2d_device_mesh()
