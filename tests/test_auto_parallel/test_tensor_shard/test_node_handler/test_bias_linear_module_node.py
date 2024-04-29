import pytest
import torch

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler import LinearFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, run_on_environment_flag, spawn
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


class LinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        return x


def check_linear_module_handler(rank, world_size, port, bias):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = LinearModule(16, 32, bias=bias).cuda()

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    input = torch.rand(4, 4, 4, 16).cuda()
    # the index of linear node in computation graph
    node_index = 3
    # strategy number of linear node
    strategy_number = 24
    # construct input args
    input_args = [input]
    # construct meta arg names
    meta_arg_names = ["x"]
    numerical_test_for_node_strategy(
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=input_args,
        meta_arg_names=meta_arg_names,
        node_type="bias_module",
    )

    tracer = ColoTracer(bias_addition_split=True)
    meta_args = {"x": torch.rand(4, 4, 4, 16).to("meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())

    linear_mod_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    handler = LinearFunctionHandler(node=linear_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)
    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["input"].name == "x"
    assert mapping["input"].data.shape == torch.Size([4, 4, 4, 16])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([64, 16])

    assert mapping["other"].name == "linear_weight"
    assert mapping["other"].data.shape == torch.Size([32, 16])
    assert mapping["other"].type == OperationDataType.PARAM
    assert mapping["other"].logical_shape == torch.Size([16, 32])

    assert "bias" not in mapping

    assert mapping["output"].name == "linear"
    assert mapping["output"].data.shape == torch.Size([4, 4, 4, 32])
    assert mapping["output"].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # SS = SR x RS
    assert "S0S1 = S0R x RS1_0" in strategy_name_list
    assert "S0S1 = S0R x RS1_1" in strategy_name_list
    assert "S0S1 = S0R x RS1_2" in strategy_name_list
    assert "S1S0 = S1R x RS0_0" in strategy_name_list
    assert "S1S0 = S1R x RS0_1" in strategy_name_list
    assert "S1S0 = S1R x RS0_2" in strategy_name_list

    # SR = SS x SR
    assert "S0R = S0S1 x S1R_0" in strategy_name_list
    assert "S0R = S0S1 x S1R_1" in strategy_name_list
    assert "S0R = S0S1 x S1R_2" in strategy_name_list
    assert "S1R = S1S0 x S0R_0" in strategy_name_list
    assert "S1R = S1S0 x S0R_1" in strategy_name_list
    assert "S1R = S1S0 x S0R_2" in strategy_name_list

    # RS = RS x SS
    assert "RS0 = RS1 x S1S0" in strategy_name_list
    assert "RS1 = RS0 x S0S1" in strategy_name_list

    # RR = RS x SR
    assert "RR = RS0 x S0R" in strategy_name_list
    assert "RR = RS1 x S1R" in strategy_name_list

    # RS= RR x RS
    assert "RS0 = RR x RS0" in strategy_name_list
    assert "RS1 = RR x RS1" in strategy_name_list

    # S01R = S01R x RR
    assert "S01R = S01R x RR_0" in strategy_name_list
    assert "S01R = S01R x RR_1" in strategy_name_list
    assert "S01R = S01R x RR_2" in strategy_name_list

    # RR = RS01 x S01R
    assert "RR = RS01 x S01R" in strategy_name_list

    # RS01 = RR x RS01
    assert "RS01 = RR x RS01" in strategy_name_list

    # RR = RR x RR
    assert "RR = RR x RR" in strategy_name_list

    for strategy in strategies_vector:
        strategy: ShardingStrategy
        input_sharding_spec = strategy.get_sharding_spec_by_name("x")
        weight_sharding_spec = strategy.get_sharding_spec_by_name("linear_weight")
        output_sharding_spec = strategy.get_sharding_spec_by_name("linear")

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[:-1] == output_sharding_spec.sharding_sequence[:-1]
        assert weight_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert weight_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[-1]


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_linear_handler(bias=True):
    spawn(check_linear_module_handler, bias=bias)


if __name__ == "__main__":
    test_linear_handler()
