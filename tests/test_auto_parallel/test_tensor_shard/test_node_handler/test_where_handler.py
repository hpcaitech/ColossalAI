import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.where_handler import WhereHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import clear_cache_before_run


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, condition, x, y):
        output = torch.where(condition, x, y)
        return output


@pytest.mark.skip("ShapeProp is not compatible with PyTorch 1.11.0")
@clear_cache_before_run()
def test_where_handler():
    model = ConvModel()
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %condition : torch.Tensor [#users=1] = placeholder[target=condition]
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %y : torch.Tensor [#users=1] = placeholder[target=y]
    #     %where : [#users=1] = call_function[target=torch.where](args = (%condition, %x, %y), kwargs = {})
    #     return where
    meta_args = {
        "condition": torch.rand(4, 4, 64, 64).to("meta"),
        "x": torch.rand(4, 1, 64, 64).to("meta"),
        "y": torch.rand(1, 4, 64, 64).to("meta"),
    }
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    where_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(where_node)

    # build handler
    handler = WhereHandler(node=where_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping, _ = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["condition"].name == "condition"
    assert mapping["condition"].data.is_meta
    assert mapping["condition"].data.shape == torch.Size([4, 4, 64, 64])
    assert mapping["condition"].type == OperationDataType.ARG
    assert mapping["condition"].logical_shape == torch.Size([4, 4, 64, 64])

    assert mapping["x"].name == "x"
    assert mapping["x"].data.is_meta
    assert mapping["x"].data.shape == torch.Size([4, 1, 64, 64])
    assert mapping["x"].type == OperationDataType.ARG
    assert mapping["x"].logical_shape == torch.Size([4, 4, 64, 64])

    assert mapping["y"].name == "y"
    assert mapping["y"].data.is_meta
    assert mapping["y"].data.shape == torch.Size([1, 4, 64, 64])
    assert mapping["y"].type == OperationDataType.ARG
    assert mapping["y"].logical_shape == torch.Size([4, 4, 64, 64])

    assert mapping["output"].name == "where"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size([4, 4, 64, 64])
    assert mapping["output"].type == OperationDataType.OUTPUT

    handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
    # 4*3 + 4*3/2*2 + 1
    assert len(strategy_name_list) == 25


if __name__ == "__main__":
    test_where_handler()
