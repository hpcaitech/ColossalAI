import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.output_handler import OutputHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import clear_cache_before_run, parameterize


class OutputModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x * 2
        return x, y


@pytest.mark.skip("ShapeProp is not compatible with PyTorch 1.11.0")
@parameterize("output_option", ["distributed", "replicated"])
@clear_cache_before_run()
def test_output_handler(output_option):
    model = OutputModel()
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %x : torch.Tensor [#users=2] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     return (x, mul)
    meta_args = {"x": torch.rand(4, 4, 64, 64).to("meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    output_node = list(graph.nodes)[2]
    output_strategies_vector = StrategiesVector(output_node)

    # build handler
    output_handler = OutputHandler(
        node=output_node,
        device_mesh=device_mesh,
        strategies_vector=output_strategies_vector,
        output_option=output_option,
    )

    output_handler.register_strategy(compute_resharding_cost=False)
    # check operation data mapping
    mapping = output_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping["output"].name == "output"
    assert mapping["output"].type == OperationDataType.OUTPUT
    strategy_name_list = [val.name for val in output_handler.strategies_vector]
    if output_option == "distributed":
        assert "Distributed Output" in strategy_name_list
    else:
        assert "Replica Output" in strategy_name_list


if __name__ == "__main__":
    test_output_handler()
