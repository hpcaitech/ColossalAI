import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.unary_elementwise_handler import UnaryElementwiseHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.tracer.meta_patch.patched_module import linear
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class ReLuModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.act = torch.nn.ReLU()

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other)
        relu_node = self.act(conv_node)
        return relu_node


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_elementwise_handler():
    model = ReLuModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %other : torch.Tensor [#users=1] = placeholder[target=other]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {})
    #     %act : [#users=1] = call_module[target=act](args = (%conv2d,), kwargs = {})
    #     return act
    graph = tracer.trace(model,
                         meta_args={
                             "input": torch.rand(4, 4, 64, 64).to('meta'),
                             "other": torch.rand(4, 16, 3, 3).to('meta'),
                         })
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    conv_mod_node = list(graph.nodes)[2]
    relu_mod_node = list(graph.nodes)[3]
    relu_strategies_vector = StrategiesVector(relu_mod_node)
    conv_strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    conv_handler = ConvFunctionHandler(node=conv_mod_node,
                                       device_mesh=device_mesh,
                                       strategies_vector=conv_strategies_vector)
    conv_handler.register_strategy(compute_resharding_cost=False)
    setattr(conv_mod_node, 'strategies_vector', conv_strategies_vector)
    relu_handler = UnaryElementwiseHandler(node=relu_mod_node,
                                           device_mesh=device_mesh,
                                           strategies_vector=relu_strategies_vector)

    relu_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = relu_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['input'].name == "conv2d"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4, 62, 62])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4, 62, 62])

    assert mapping['output'].name == "act"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 4, 62, 62])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # getitem is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(relu_strategies_vector) == len(conv_strategies_vector)


if __name__ == '__main__':
    test_elementwise_handler()
