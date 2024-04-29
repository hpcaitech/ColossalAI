import pytest
import torch
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import BMMFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use, run_on_environment_flag, spawn
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


class AddBMMTensorMethodModule(nn.Module):
    def __init__(self, using_kwargs):
        super().__init__()
        self.using_kwargs = using_kwargs

    def forward(self, bias, x1, x2):
        if self.using_kwargs:
            output = bias.addbmm(x1, x2, alpha=2, beta=3)
        else:
            output = bias.addbmm(x1, x2)
        return output


class AddBMMTorchFunctionModule(nn.Module):
    def __init__(self, using_kwargs):
        super().__init__()
        self.using_kwargs = using_kwargs

    def forward(self, bias, x1, x2):
        if self.using_kwargs:
            output = torch.addbmm(bias, x1, x2, alpha=2, beta=3)
        else:
            output = torch.addbmm(bias, x1, x2)
        return output


def check_2d_device_mesh(rank, world_size, port, module, bias_shape, using_kwargs):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = module(using_kwargs).cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    x1 = torch.rand(4, 8, 16).cuda()
    x2 = torch.rand(4, 16, 8).cuda()
    bias = torch.rand(bias_shape).cuda()
    # the index of addbmm node in computation graph
    node_index = 3
    # strategy number of addbmm node on 2d device mesh
    strategy_number = 7
    # construct input args
    input_args = [bias, x1, x2]
    # construct meta arg names
    meta_arg_names = ["bias", "x1", "x2"]
    numerical_test_for_node_strategy(
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=input_args,
        meta_arg_names=meta_arg_names,
    )
    tracer = ColoTracer()
    # graph():
    #     %bias : torch.Tensor [#users=1] = placeholder[target=bias]
    #     %x1 : torch.Tensor [#users=1] = placeholder[target=x1]
    #     %x2 : torch.Tensor [#users=1] = placeholder[target=x2]
    #     %bmm : [#users=1] = call_function[target=torch.bmm](args = (%x1, %x2), kwargs = {})
    #     %sum_1 : [#users=1] = call_function[target=torch.sum](args = (%bmm, 0), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%sum_1, %bias), kwargs = {})
    #     return add
    graph = tracer.trace(
        model,
        meta_args={
            "bias": torch.rand(*bias_shape).to("meta"),
            "x1": torch.rand(4, 8, 16).to("meta"),
            "x2": torch.rand(4, 16, 8).to("meta"),
        },
    )
    ColoGraphModule(model, graph)

    bmm_mod_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(bmm_mod_node)

    # build handler
    handler = BMMFunctionHandler(node=bmm_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["input"].name == "x1"
    assert mapping["input"].data.is_meta
    assert mapping["input"].data.shape == torch.Size([4, 8, 16])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([4, 8, 16])

    assert mapping["other"].name == "x2"
    assert mapping["other"].data.is_meta
    assert mapping["other"].data.shape == torch.Size([4, 16, 8])
    assert mapping["other"].type == OperationDataType.ARG
    assert mapping["other"].logical_shape == torch.Size([4, 16, 8])

    assert mapping["output"].name == "bmm"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size([4, 8, 8])
    assert mapping["output"].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
    for name in strategy_name_list:
        print(name)
    # one batch dim
    assert "Sb0 = Sb0 x Sb0" not in strategy_name_list

    # two batch dim
    assert "Sb01 = Sb01 x Sb01" in strategy_name_list

    # SbSi = SbSi x Sb
    assert "Sb0Si1 = Sb0Si1 x Sb0" in strategy_name_list
    assert "Sb1Si0 = Sb1Si0 x Sb1" in strategy_name_list

    # SbSj = SbR x SbSj
    assert "Sb0Sj1 = Sb0R x Sb0Sj1" in strategy_name_list
    assert "Sb1Sj0 = Sb1R x Sb1Sj0" in strategy_name_list

    # SbR = SbSk x SbSk
    assert "Sb0R = Sb0Sk1 x Sb0Sk1" in strategy_name_list
    assert "Sb1R = Sb1Sk0 x Sb1Sk0" in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name("x1")
        other_sharding_spec = strategy.get_sharding_spec_by_name("x2")
        output_sharding_spec = strategy.get_sharding_spec_by_name("bmm")

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[0]
        assert other_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert other_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


def check_1d_device_mesh(rank, module, bias_shape, using_kwargs, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (1, 4)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    model = module(using_kwargs).cuda()
    x1 = torch.rand(4, 8, 16).cuda()
    x2 = torch.rand(4, 16, 8).cuda()
    bias = torch.rand(bias_shape).cuda()
    # the index of addbmm node in computation graph
    node_index = 3
    # strategy number of addbmm node on 2d device mesh
    strategy_number = 1
    # construct input args
    input_args = [bias, x1, x2]
    # construct meta arg names
    meta_arg_names = ["bias", "x1", "x2"]
    numerical_test_for_node_strategy(
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=input_args,
        meta_arg_names=meta_arg_names,
    )

    tracer = ColoTracer()
    # graph():
    #     %bias : torch.Tensor [#users=1] = placeholder[target=bias]
    #     %x1 : torch.Tensor [#users=1] = placeholder[target=x1]
    #     %x2 : torch.Tensor [#users=1] = placeholder[target=x2]
    #     %bmm : [#users=1] = call_function[target=torch.bmm](args = (%x1, %x2), kwargs = {})
    #     %sum_1 : [#users=1] = call_function[target=torch.sum](args = (%bmm, 0), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%sum_1, %bias), kwargs = {})
    #     return add
    graph = tracer.trace(
        model,
        meta_args={
            "bias": torch.rand(*bias_shape).to("meta"),
            "x1": torch.rand(4, 8, 16).to("meta"),
            "x2": torch.rand(4, 16, 8).to("meta"),
        },
    )
    ColoGraphModule(model, graph)
    bmm_mod_node = list(graph.nodes)[3]
    strategies_vector = StrategiesVector(bmm_mod_node)

    # build handler
    handler = BMMFunctionHandler(node=bmm_mod_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["input"].name == "x1"
    assert mapping["input"].data.is_meta
    assert mapping["input"].data.shape == torch.Size([4, 8, 16])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([4, 8, 16])

    assert mapping["other"].name == "x2"
    assert mapping["other"].data.is_meta
    assert mapping["other"].data.shape == torch.Size([4, 16, 8])
    assert mapping["other"].type == OperationDataType.ARG
    assert mapping["other"].logical_shape == torch.Size([4, 16, 8])

    assert mapping["output"].name == "bmm"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size([4, 8, 8])
    assert mapping["output"].type == OperationDataType.OUTPUT

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]
    assert len(strategy_name_list) == 1
    # one batch dim
    assert "Sb0 = Sb0 x Sb0" in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name("x1")
        other_sharding_spec = strategy.get_sharding_spec_by_name("x2")
        output_sharding_spec = strategy.get_sharding_spec_by_name("bmm")

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[0]
        assert other_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[-1]
        assert other_sharding_spec.sharding_sequence[-1] == output_sharding_spec.sharding_sequence[-1]


@pytest.mark.skip("skip due to bias cases not ready")
@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@parameterize("module", [AddBMMTorchFunctionModule, AddBMMTensorMethodModule])
@parameterize("bias_shape", [[8], [1, 8], [8, 8]])
@parameterize("using_kwargs", [True, False])
@rerun_if_address_is_in_use()
def test_2d_device_mesh(module, bias_shape, using_kwargs):
    spawn(
        check_2d_device_mesh,
        4,
        module=module,
        bias_shape=bias_shape,
        using_kwargs=using_kwargs,
    )


@pytest.mark.skip("skip due to bias cases not ready")
@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@parameterize("module", [AddBMMTorchFunctionModule, AddBMMTensorMethodModule])
@parameterize("bias_shape", [[8], [1, 8], [8, 8]])
@parameterize("using_kwargs", [True, False])
@rerun_if_address_is_in_use()
def test_1d_device_mesh(module, bias_shape, using_kwargs):
    spawn(
        check_1d_device_mesh,
        4,
        module=module,
        bias_shape=bias_shape,
        using_kwargs=using_kwargs,
    )


if __name__ == "__main__":
    test_1d_device_mesh()
    test_2d_device_mesh()
