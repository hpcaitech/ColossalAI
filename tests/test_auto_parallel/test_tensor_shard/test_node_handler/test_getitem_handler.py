from colossalai.fx.tracer.meta_patch.patched_module import linear
import torch
import torch.nn as nn
from colossalai.fx import ColoTracer, ColoGraphModule
from colossalai.auto_parallel.solver.node_handler.getitem_handler import GetItemHandler
from colossalai.auto_parallel.solver.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.solver.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh


class GetItemModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other)
        x = conv_node[1]
        return x


def test_getitem_function_handler():
    model = GetItemModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %other : torch.Tensor [#users=1] = placeholder[target=other]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {})
    #     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%conv2d, 1), kwargs = {})
    #     return getitem
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
    getitem_mod_node = list(graph.nodes)[3]
    getitem_strategies_vector = StrategiesVector(getitem_mod_node)
    conv_strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    conv_handler = ConvFunctionHandler(node=conv_mod_node,
                                       device_mesh=device_mesh,
                                       strategies_vector=conv_strategies_vector)
    conv_handler.register_strategy(compute_resharding_cost=False)
    setattr(conv_mod_node, 'strategies_vector', conv_strategies_vector)
    getitem_handler = GetItemHandler(node=getitem_mod_node,
                                     device_mesh=device_mesh,
                                     strategies_vector=getitem_strategies_vector)

    getitem_handler.register_strategy(compute_resharding_cost=False)
    # check operation data mapping
    mapping = getitem_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['input'].name == "conv2d"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([4, 4, 62, 62])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 4, 62, 62])

    assert mapping['index'].name == "index"
    assert isinstance(mapping['index'].data, int)
    assert mapping['index'].type == OperationDataType.ARG

    assert mapping['output'].name == "getitem"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([4, 62, 62])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # getitem is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(getitem_strategies_vector) == len(conv_strategies_vector)


if __name__ == '__main__':
    test_getitem_function_handler()
