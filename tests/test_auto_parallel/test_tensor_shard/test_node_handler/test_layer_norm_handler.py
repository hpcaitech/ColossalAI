import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.layer_norm_handler import LayerNormModuleHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


def check_ln_module_handler(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = nn.Sequential(nn.LayerNorm(16)).cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    input = torch.rand(4, 16).cuda()
    # the index of bn node in computation graph
    node_index = 1
    # the total number of ln strategies
    strategy_number = 4
    # construct input args
    input_args = [input]
    # construct meta arg names
    meta_arg_names = ["input"]
    numerical_test_for_node_strategy(
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=input_args,
        meta_arg_names=meta_arg_names,
    )
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %_0 : [#users=1] = call_module[target=0](args = (%input_1,), kwargs = {})
    #     return _0
    meta_args = {"input": torch.rand(4, 16).to("meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())

    ln_mod_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(ln_mod_node)

    # build handler
    handler = LayerNormModuleHandler(node=ln_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["input"].name == "input_1"
    assert mapping["input"].data.shape == torch.Size([4, 16])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([4, 16])

    assert mapping["other"].name == "weight"
    assert mapping["other"].data.shape == torch.Size([16])
    assert mapping["other"].type == OperationDataType.PARAM
    assert mapping["other"].logical_shape == torch.Size([16])

    assert mapping["bias"].name == "bias"
    assert mapping["bias"].data.shape == torch.Size([16])
    assert mapping["bias"].type == OperationDataType.PARAM
    assert mapping["bias"].logical_shape == torch.Size([16])

    assert mapping["output"].name == "_0"
    assert mapping["output"].data.shape == torch.Size([4, 16])
    assert mapping["output"].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # SR = SR x R
    assert "[S0, R] = [S0, R] x [R]" in strategy_name_list
    assert "[S1, R] = [S1, R] x [R]" in strategy_name_list

    # RR = RR x R
    assert "RR = RR x R" in strategy_name_list

    # S01R = S01R x R
    assert "[S01, R] = [S01, R] x [R]" in strategy_name_list


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_ln_module_handler():
    spawn(check_ln_module_handler, 4)


if __name__ == "__main__":
    test_ln_module_handler()
