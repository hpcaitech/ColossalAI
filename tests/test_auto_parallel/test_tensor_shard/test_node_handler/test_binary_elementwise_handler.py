import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import BinaryElementwiseHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing import parameterize


@parameterize('op', [torch.add])
@parameterize('other_dim', [1, 2])
def test_binary_elementwise_handler_with_tensor(op, other_dim):

    class BinaryElementwiseOpModel(nn.Module):

        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x1, x2):
            out = self.op(x1, x2)
            return out

    model = BinaryElementwiseOpModel(op)
    tracer = ColoTracer()

    meta_args = {'x1': torch.rand(4, 4).to('meta'), 'x2': torch.rand([4] * other_dim).to('meta')}
    graph = tracer.trace(model, meta_args=meta_args)
    print(graph)
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    op_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(op_node)

    # build handler
    handler = BinaryElementwiseHandler(node=op_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping['input'].name == "x1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4])

    assert mapping['other'].name == "x2"
    assert mapping['other'].data.is_meta
    assert mapping['other'].data.shape == torch.Size([4] * other_dim)
    assert mapping['other'].type == OperationDataType.ARG
    assert mapping['other'].logical_shape == torch.Size([4, 4])

    assert mapping['output'].name == str(op_node)
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 4])
    assert mapping['output'].type == OperationDataType.OUTPUT
    assert mapping['output'].logical_shape == torch.Size([4, 4])

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # one strategy will be converted to different physical sharding spec
    assert len(strategy_name_list) == 9

    # check if the sharding strategy is correct
    assert '[S0, S1] = [S0, S1] <binary-elementwise-op> [S0, S1]' in strategy_name_list
    assert '[S1, S0] = [S1, S0] <binary-elementwise-op> [S1, S0]' in strategy_name_list
    assert '[S01, R] = [S01, R] <binary-elementwise-op> [S01, R]' in strategy_name_list
    assert '[R, S01] = [R, S01] <binary-elementwise-op> [R, S01]' in strategy_name_list
    assert '[S0, R] = [S0, R] <binary-elementwise-op> [S0, R]' in strategy_name_list
    assert '[R, S0] = [R, S0] <binary-elementwise-op> [R, S0]' in strategy_name_list
    assert '[S1, R] = [S1, R] <binary-elementwise-op> [S1, R]' in strategy_name_list
    assert '[R, S1] = [R, S1] <binary-elementwise-op> [R, S1]' in strategy_name_list
    assert '[R, R] = [R, R] <binary-elementwise-op> [R, R]' in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name('x1')
        other_sharding_spec = strategy.get_sharding_spec_by_name('x2')
        output_sharding_spec = strategy.get_sharding_spec_by_name(str(op_node))

        # make sure the sharding spec is the same for input and output
        assert input_sharding_spec.sharding_sequence == output_sharding_spec.sharding_sequence

        # since the dim of the other can change, we make sure at least its last dim sharding is the same
        if len(other_sharding_spec.sharding_sequence) == 2:
            assert input_sharding_spec.sharding_sequence == other_sharding_spec.sharding_sequence
        elif len(other_sharding_spec.sharding_sequence) == 1:
            assert input_sharding_spec.sharding_sequence[-1] == other_sharding_spec.sharding_sequence[-1]


@parameterize('op', [torch.add])
@parameterize('other', [1, 2])
def test_binary_elementwise_handler_with_int(op, other):

    class BinaryElementwiseOpModel(nn.Module):

        def __init__(self, op, const):
            super().__init__()
            self.op = op
            self.const = const

        def forward(self, x1):
            out = self.op(x1, self.const)
            return out

    model = BinaryElementwiseOpModel(op, other)
    tracer = ColoTracer()

    meta_args = {'x1': torch.rand(4, 4).to('meta')}
    graph = tracer.trace(model, meta_args=meta_args)
    print(graph)
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    op_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(op_node)

    # build handler
    handler = BinaryElementwiseHandler(node=op_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    assert mapping['input'].name == "x1"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4])

    assert mapping['output'].name == str(op_node)
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 4])
    assert mapping['output'].type == OperationDataType.OUTPUT
    assert mapping['output'].logical_shape == torch.Size([4, 4])

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # one strategy will be converted to different physical sharding spec
    assert len(strategy_name_list) == 9

    # check if the sharding strategy is correct
    assert '[S0, S1] = [S0, S1] <binary-elementwise-op> [S0, S1]' in strategy_name_list
    assert '[S1, S0] = [S1, S0] <binary-elementwise-op> [S1, S0]' in strategy_name_list
    assert '[S01, R] = [S01, R] <binary-elementwise-op> [S01, R]' in strategy_name_list
    assert '[R, S01] = [R, S01] <binary-elementwise-op> [R, S01]' in strategy_name_list
    assert '[S0, R] = [S0, R] <binary-elementwise-op> [S0, R]' in strategy_name_list
    assert '[R, S0] = [R, S0] <binary-elementwise-op> [R, S0]' in strategy_name_list
    assert '[S1, R] = [S1, R] <binary-elementwise-op> [S1, R]' in strategy_name_list
    assert '[R, S1] = [R, S1] <binary-elementwise-op> [R, S1]' in strategy_name_list
    assert '[R, R] = [R, R] <binary-elementwise-op> [R, R]' in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name('x1')
        output_sharding_spec = strategy.get_sharding_spec_by_name(str(op_node))

        # make sure the sharding spec is the same for input and output
        assert input_sharding_spec.sharding_sequence == output_sharding_spec.sharding_sequence


if __name__ == '__main__':
    test_binary_elementwise_handler_with_tensor()
    test_binary_elementwise_handler_with_int()
