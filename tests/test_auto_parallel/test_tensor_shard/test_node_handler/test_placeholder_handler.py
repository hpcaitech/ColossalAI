import torch
import torch.nn as nn
from colossalai.fx import ColoTracer, ColoGraphModule
from colossalai.auto_parallel.solver.node_handler.placeholder_handler import PlacehodlerHandler
from colossalai.auto_parallel.solver.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh


class PlaceholderModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def test_placeholder_handler():
    model = PlaceholderModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     return input_1
    graph = tracer.trace(model, meta_args={
        "input": torch.rand(4, 4, 64, 64).to('meta'),
    })
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    placeholder_node = list(graph.nodes)[0]
    placeholder_strategies_vector = StrategiesVector(placeholder_node)

    # build handler
    placeholder_handler = PlacehodlerHandler(node=placeholder_node,
                                             device_mesh=device_mesh,
                                             strategies_vector=placeholder_strategies_vector)

    placeholder_handler.register_strategy(compute_resharding_cost=False)
    # check operation data mapping
    mapping = placeholder_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['output'].name == "input_1"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size((4, 4, 64, 64))
    assert mapping['output'].type == OperationDataType.OUTPUT
    strategy_name_list = [val.name for val in placeholder_handler.strategies_vector]
    assert "Replica Placeholder" in strategy_name_list


if __name__ == '__main__':
    test_placeholder_handler()
