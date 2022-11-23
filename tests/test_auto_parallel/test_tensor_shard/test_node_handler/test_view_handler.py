import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.experimental import ViewHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class ViewModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other)
        reshape_node = conv_node.view(32, 4, 32, 32, 4)
        return reshape_node


def test_view_handler():
    model = ViewModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %other : torch.Tensor [#users=1] = placeholder[target=other]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {})
    #     %view : [#users=1] = call_method[target=view](args = (%conv2d, 2, -1), kwargs = {})
    #     return view
    graph = tracer.trace(model,
                         meta_args={
                             "input": torch.rand(8, 8, 66, 66).to('meta'),
                             "other": torch.rand(16, 8, 3, 3).to('meta'),
                         })
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    conv_mod_node = list(graph.nodes)[2]
    view_node = list(graph.nodes)[3]
    view_strategies_vector = StrategiesVector(view_node)
    conv_strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    conv_handler = ConvFunctionHandler(node=conv_mod_node,
                                       device_mesh=device_mesh,
                                       strategies_vector=conv_strategies_vector)
    conv_handler.register_strategy(compute_resharding_cost=False)
    setattr(conv_mod_node, 'strategies_vector', conv_strategies_vector)
    view_handler = ViewHandler(node=view_node, device_mesh=device_mesh, strategies_vector=view_strategies_vector)

    view_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = view_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['input'].name == "conv2d"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([8, 16, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([8, 16, 64, 64])

    assert mapping['output'].name == "view"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([32, 4, 32, 32, 4])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # reshape handler is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(view_strategies_vector) == len(conv_strategies_vector)
    strategy_name_list = [strategy.name for strategy in view_strategies_vector]
    assert '[S0, S1, R, R] -> FULLY REPLICATED_0' in strategy_name_list
    assert '[S1, S0, R, R] -> FULLY REPLICATED_1' in strategy_name_list
    assert '[S0, R, R, R] -> [S0, R, R, R, R]_2' in strategy_name_list
    assert '[S1, R, R, R] -> [S1, R, R, R, R]_3' in strategy_name_list
    assert '[S0, R, R, R] -> [S0, R, R, R, R]_4' in strategy_name_list
    assert '[S1, R, R, R] -> [S1, R, R, R, R]_5' in strategy_name_list
    assert '[R, S1, R, R] -> FULLY REPLICATED_6' in strategy_name_list
    assert '[R, S0, R, R] -> FULLY REPLICATED_7' in strategy_name_list
    assert '[R, R, R, R] -> [R, R, R, R, R]_8' in strategy_name_list
    assert '[R, R, R, R] -> [R, R, R, R, R]_9' in strategy_name_list
    assert '[R, S0, R, R] -> FULLY REPLICATED_10' in strategy_name_list
    assert '[R, S1, R, R] -> FULLY REPLICATED_11' in strategy_name_list
    assert '[R, R, R, R] -> [R, R, R, R, R]_12' in strategy_name_list
    assert '[S01, R, R, R] -> [S01, R, R, R, R]_13' in strategy_name_list
    assert '[R, R, R, R] -> [R, R, R, R, R]_14' in strategy_name_list
    assert '[R, S01, R, R] -> FULLY REPLICATED_15' in strategy_name_list


if __name__ == '__main__':
    test_view_handler()
