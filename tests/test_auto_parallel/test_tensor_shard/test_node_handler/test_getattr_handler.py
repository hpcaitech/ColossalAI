import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.getattr_handler import GetattrHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer


class GetattrModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, 3, padding=1, bias=False)

    def forward(self, input):
        weight = self.conv.weight
        return weight


def test_getattr_handler():
    model = GetattrModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=0] = placeholder[target=input]
    #     %conv_weight : [#users=1] = get_attr[target=conv.weight]
    #     return conv_weight
    graph = tracer.trace(model, meta_args={'input': torch.rand(4, 4, 64, 64).to('meta')})
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    getattr_node = list(graph.nodes)[1]
    getattr_strategies_vector = StrategiesVector(getattr_node)

    # build handler
    getattr_handler = GetattrHandler(node=getattr_node,
                                     device_mesh=device_mesh,
                                     strategies_vector=getattr_strategies_vector)

    getattr_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = getattr_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['output'].name == "conv_weight"
    assert mapping['output'].data.shape == torch.Size((16, 4, 3, 3))
    assert mapping['output'].type == OperationDataType.OUTPUT
    strategy_name_list = [val.name for val in getattr_handler.strategies_vector]
    assert 'get_attr [S0, S1, R, R]' in strategy_name_list
    assert 'get_attr [S1, S0, R, R]' in strategy_name_list
    assert 'get_attr [S01, R, R, R]' in strategy_name_list
    assert 'get_attr [R, S01, R, R]' in strategy_name_list
    assert 'get_attr [S0, R, R, R]' in strategy_name_list
    assert 'get_attr [R, S0, R, R]' in strategy_name_list
    assert 'get_attr [S1, R, R, R]' in strategy_name_list
    assert 'get_attr [R, S1, R, R]' in strategy_name_list
    assert 'get_attr [R, R, R, R]' in strategy_name_list


if __name__ == '__main__':
    test_getattr_handler()
