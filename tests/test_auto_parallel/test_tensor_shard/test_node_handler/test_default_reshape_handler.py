import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler import DefaultReshapeHandler
from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import clear_cache_before_run, run_on_environment_flag


class ReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other)
        reshape_node = conv_node.view(2, -1)
        return reshape_node


@run_on_environment_flag(name="AUTO_PARALLEL")
@clear_cache_before_run()
def test_reshape_handler():
    model = ReshapeModel()
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %other : torch.Tensor [#users=1] = placeholder[target=other]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {})
    #     %view : [#users=1] = call_method[target=view](args = (%conv2d, 2, -1), kwargs = {})
    #     return view
    meta_args = {
        "input": torch.rand(4, 4, 64, 64).to("meta"),
        "other": torch.rand(16, 4, 3, 3).to("meta"),
    }
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    conv_mod_node = list(graph.nodes)[2]
    reshape_node = list(graph.nodes)[3]
    reshape_strategies_vector = StrategiesVector(reshape_node)
    conv_strategies_vector = StrategiesVector(conv_mod_node)

    # build handler
    conv_handler = ConvFunctionHandler(
        node=conv_mod_node, device_mesh=device_mesh, strategies_vector=conv_strategies_vector
    )
    conv_handler.register_strategy(compute_resharding_cost=False)
    setattr(conv_mod_node, "strategies_vector", conv_strategies_vector)
    reshape_handler = DefaultReshapeHandler(
        node=reshape_node, device_mesh=device_mesh, strategies_vector=reshape_strategies_vector
    )

    reshape_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = reshape_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping["input"].name == "conv2d"
    assert mapping["input"].data.is_meta
    assert mapping["input"].data.shape == torch.Size([4, 16, 62, 62])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([4, 16, 62, 62])

    assert mapping["output"].name == "view"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size([2, 123008])
    assert mapping["output"].type == OperationDataType.OUTPUT

    # reshape handler is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(reshape_strategies_vector) == len(conv_strategies_vector)


if __name__ == "__main__":
    test_reshape_handler()
