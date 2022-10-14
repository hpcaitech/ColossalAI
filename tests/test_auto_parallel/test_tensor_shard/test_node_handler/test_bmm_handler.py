import pytest
import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.dot_handler import \
    BMMFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (OperationData, OperationDataType, StrategiesVector)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class BMMTensorMethodModule(nn.Module):

    def forward(self, x1, x2):
        return x1.bmm(x2)


class BMMTorchFunctionModule(nn.Module):

    def forward(self, x1, x2):
        return torch.bmm(x1, x2)


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.parametrize('module', [BMMTensorMethodModule, BMMTorchFunctionModule])
def test_2d_device_mesh(module):

    model = module()
    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             "x1": torch.rand(4, 8, 16).to('meta'),
                             'x2': torch.rand(4, 16, 8).to('meta')
                         })
    print(graph)
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    linear_mod_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = BMMFunctionHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

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

    assert mapping['output'].name == "bmm"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 8, 8])
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


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.parametrize('module', [BMMTensorMethodModule, BMMTorchFunctionModule])
def test_1d_device_mesh(module):
    model = module()
    tracer = ColoTracer()
    graph = tracer.trace(model,
                         meta_args={
                             "x1": torch.rand(4, 8, 16).to('meta'),
                             'x2': torch.rand(4, 16, 8).to('meta')
                         })
    print(graph)
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (1, 4)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    linear_mod_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = BMMFunctionHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

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

    assert mapping['output'].name == "bmm"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 8, 8])
    assert mapping['output'].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
    assert len(strategy_name_list) == 1
    # one batch dim
    assert 'Sb0 = Sb0 x Sb0' in strategy_name_list


if __name__ == '__main__':
    test_1d_device_mesh(BMMTensorMethodModule)
    test_1d_device_mesh(BMMTorchFunctionModule)
    test_2d_device_mesh(BMMTensorMethodModule)
    test_2d_device_mesh(BMMTorchFunctionModule)
