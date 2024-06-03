import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.tensor_constructor_handler import TensorConstructorHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing import clear_cache_before_run, run_on_environment_flag


class TensorConstructorModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        arange_node = torch.arange(x.size()[0])
        x = x + arange_node
        return x


@run_on_environment_flag(name="AUTO_PARALLEL")
@clear_cache_before_run()
def test_where_handler():
    model = TensorConstructorModel()
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %x : torch.Tensor [#users=2] = placeholder[target=x]
    #     %size : [#users=1] = call_method[target=size](args = (%x,), kwargs = {})
    #     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%size, 0), kwargs = {})
    #     %arange : [#users=1] = call_function[target=torch.arange](args = (%getitem,), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%x, %arange), kwargs = {})
    #     return add
    meta_args = {"x": torch.rand(10).to("meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    arange_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(arange_node)

    # build handler
    handler = TensorConstructorHandler(node=arange_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["output"].name == "arange"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size([10])
    assert mapping["output"].type == OperationDataType.OUTPUT

    handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    assert "Replica Tensor Constructor" in strategy_name_list


if __name__ == "__main__":
    test_where_handler()
