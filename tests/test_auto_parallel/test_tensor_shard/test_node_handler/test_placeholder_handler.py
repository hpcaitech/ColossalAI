import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.placeholder_handler import PlaceholderHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import clear_cache_before_run, parameterize


class PlaceholderModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


@pytest.mark.skip("ShapeProp is not compatible with PyTorch 1.11.0")
@parameterize("placeholder_option", ["distributed", "replicated"])
@clear_cache_before_run()
def test_placeholder_handler(placeholder_option):
    model = PlaceholderModel()
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     return input_1
    meta_args = {
        "input": torch.rand(4, 4, 64, 64).to("meta"),
    }
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    placeholder_node = list(graph.nodes)[0]
    placeholder_strategies_vector = StrategiesVector(placeholder_node)
    # build handler
    placeholder_handler = PlaceholderHandler(
        node=placeholder_node,
        device_mesh=device_mesh,
        strategies_vector=placeholder_strategies_vector,
        placeholder_option=placeholder_option,
    )

    placeholder_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = placeholder_handler.get_operation_data_mapping()

    strategy = placeholder_strategies_vector[0]
    strategy_sharding_spec = strategy.get_sharding_spec_by_name(mapping["output"].name)

    if placeholder_option == "distributed":
        assert str(strategy_sharding_spec.sharding_sequence) == "[S01, R, R, R]"
    else:
        assert str(strategy_sharding_spec.sharding_sequence) == "[R, R, R, R]"

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping["output"].name == "input_1"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size((4, 4, 64, 64))
    assert mapping["output"].type == OperationDataType.OUTPUT
    strategy_name_list = [val.name for val in placeholder_handler.strategies_vector]
    if placeholder_option == "replicated":
        assert "Replica Placeholder" in strategy_name_list
    else:
        assert "Distributed Placeholder" in strategy_name_list


if __name__ == "__main__":
    test_placeholder_handler()
