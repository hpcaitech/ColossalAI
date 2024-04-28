import pytest
import torch
import torch.nn as nn

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.node_handler.embedding_handler import (
    EmbeddingFunctionHandler,
    EmbeddingModuleHandler,
)
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy

NUM_EMBEDDINGS = 16
EMBEDDING_DIMS = 32


class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dims):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dims)

    def forward(self, input):
        x = self.embedding(input)
        return x


def check_embedding_module_handler(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = EmbeddingModule(num_embeddings=NUM_EMBEDDINGS, embedding_dims=EMBEDDING_DIMS).cuda()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %embedding : [#users=1] = call_module[target=embedding](args = (%input_1,), kwargs = {})
    #     return embedding
    input = torch.rand(4, 16, 16) * NUM_EMBEDDINGS
    input = input.to(torch.int64).cuda()

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    # index of embedding node in computation graph
    node_index = 1
    # total number of embedding strategies
    strategy_number = 19
    numerical_test_for_node_strategy(
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=[input],
        meta_arg_names=["input"],
    )

    tracer = ColoTracer(bias_addition_split=True)
    meta_args = {"input": torch.randint(NUM_EMBEDDINGS, (4, 16, 16)).to("meta")}
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())
    embedding_node = list(graph.nodes)[1]
    strategies_vector = StrategiesVector(embedding_node)

    # build handler
    handler = EmbeddingModuleHandler(node=embedding_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["input"].name == "input_1"
    # assert mapping['input'].data.is_meta
    assert mapping["input"].data.shape == torch.Size([4, 16, 16])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([1024])

    assert mapping["other"].name == "weight"
    assert mapping["other"].data.shape == torch.Size([NUM_EMBEDDINGS, EMBEDDING_DIMS])
    assert mapping["other"].type == OperationDataType.PARAM
    assert mapping["other"].logical_shape == torch.Size([NUM_EMBEDDINGS, EMBEDDING_DIMS])

    assert mapping["output"].name == "embedding"
    assert mapping["output"].data.shape == torch.Size([4, 16, 16, EMBEDDING_DIMS])
    assert mapping["output"].type == OperationDataType.OUTPUT
    assert mapping["output"].logical_shape == torch.Size([1024, EMBEDDING_DIMS])

    strategies_vector = handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # RR = RR x RR
    assert "RR = R x RR" in strategy_name_list

    # SR = SR x RR
    assert "S0R = S0 x RR_0" in strategy_name_list
    assert "S0R = S0 x RR_1" in strategy_name_list
    assert "S0R = S0 x RR_2" in strategy_name_list
    assert "S1R = S1 x RR_0" in strategy_name_list
    assert "S1R = S1 x RR_1" in strategy_name_list
    assert "S1R = S1 x RR_2" in strategy_name_list

    # SS = SR x RS
    assert "S0S1 = S0 x RS1_0" in strategy_name_list
    assert "S0S1 = S0 x RS1_1" in strategy_name_list
    assert "S0S1 = S0 x RS1_2" in strategy_name_list
    assert "S1S0 = S1 x RS0_0" in strategy_name_list
    assert "S1S0 = S1 x RS0_1" in strategy_name_list
    assert "S1S0 = S1 x RS0_2" in strategy_name_list

    # RS= RR x RS
    assert "RS0 = R x RS0" in strategy_name_list
    assert "RS1 = R x RS1" in strategy_name_list

    # S01R = S01R x RR
    assert "S01R = S01 x RR_0" in strategy_name_list
    assert "S01R = S01 x RR_1" in strategy_name_list
    assert "S01R = S01 x RR_2" in strategy_name_list

    # RS01 = RR x RS01
    assert "RS01 = R x RS01" in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name("input_1")
        weight_sharding_spec = strategy.get_sharding_spec_by_name("weight")
        output_sharding_spec = strategy.get_sharding_spec_by_name("embedding")

        # make sure the sharding matches across different operation data
        assert output_sharding_spec.sharding_sequence[-1] == weight_sharding_spec.sharding_sequence[-1]
        assert input_sharding_spec.sharding_sequence == output_sharding_spec.sharding_sequence[:-1]


class EmbeddingFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, others):
        x = nn.functional.embedding(input, others)
        return x


def check_embedding_function_handler(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = EmbeddingFunction().cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    input = torch.rand(4, 16, 16) * NUM_EMBEDDINGS
    input = input.to(torch.int64).cuda()
    others = torch.rand(NUM_EMBEDDINGS, EMBEDDING_DIMS).cuda()
    input_args = [input, others]
    meta_arg_names = ["input", "others"]
    input_kwargs = {}
    # total number of embedding strategies
    strategy_number = 19
    node_index = 2
    numerical_test_for_node_strategy(
        model=model,
        device_mesh=device_mesh,
        node_index=node_index,
        strategy_number=strategy_number,
        input_args=input_args,
        meta_arg_names=meta_arg_names,
        input_kwargs=input_kwargs,
    )
    tracer = ColoTracer(bias_addition_split=True)
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %others : torch.Tensor [#users=1] = placeholder[target=others]
    #     %embedding : [#users=1] = call_function[target=torch.nn.functional.embedding](args = (%input_1, %others), kwargs = {padding_idx: None, max_norm: None, norm_type: 2.0, scale_grad_by_freq: False, sparse: False})
    #     return embedding
    meta_args = {
        "input": torch.randint(NUM_EMBEDDINGS, (4, 16, 16)).to("meta"),
        "others": torch.rand(NUM_EMBEDDINGS, EMBEDDING_DIMS).to("meta"),
    }
    graph = tracer.trace(model, meta_args=meta_args)
    gm = ColoGraphModule(model, graph)
    shape_prop_pass(gm, *meta_args.values())

    embedding_node = list(graph.nodes)[2]
    strategies_vector = StrategiesVector(embedding_node)

    # build handler
    handler = EmbeddingFunctionHandler(
        node=embedding_node, device_mesh=device_mesh, strategies_vector=strategies_vector
    )

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.logical_shape is not None
        assert op_data.data is not None

    assert mapping["input"].name == "input_1"
    assert mapping["input"].data.is_meta
    assert mapping["input"].data.shape == torch.Size([4, 16, 16])
    assert mapping["input"].type == OperationDataType.ARG
    assert mapping["input"].logical_shape == torch.Size([1024])

    assert mapping["other"].name == "others"
    assert mapping["other"].data.is_meta
    assert mapping["other"].data.shape == torch.Size([NUM_EMBEDDINGS, EMBEDDING_DIMS])
    assert mapping["other"].type == OperationDataType.ARG
    assert mapping["other"].logical_shape == torch.Size([NUM_EMBEDDINGS, EMBEDDING_DIMS])

    assert mapping["output"].name == "embedding"
    assert mapping["output"].data.is_meta
    assert mapping["output"].data.shape == torch.Size([4, 16, 16, EMBEDDING_DIMS])
    assert mapping["output"].type == OperationDataType.OUTPUT
    assert mapping["output"].logical_shape == torch.Size([1024, EMBEDDING_DIMS])

    handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # RR = RR x RR
    assert "RR = R x RR" in strategy_name_list

    # SR = SR x RR
    assert "S0R = S0 x RR_0" in strategy_name_list
    assert "S0R = S0 x RR_1" in strategy_name_list
    assert "S0R = S0 x RR_2" in strategy_name_list
    assert "S1R = S1 x RR_0" in strategy_name_list
    assert "S1R = S1 x RR_1" in strategy_name_list
    assert "S1R = S1 x RR_2" in strategy_name_list

    # SS = SR x RS
    assert "S0S1 = S0 x RS1_0" in strategy_name_list
    assert "S0S1 = S0 x RS1_1" in strategy_name_list
    assert "S0S1 = S0 x RS1_2" in strategy_name_list
    assert "S1S0 = S1 x RS0_0" in strategy_name_list
    assert "S1S0 = S1 x RS0_1" in strategy_name_list
    assert "S1S0 = S1 x RS0_2" in strategy_name_list

    # RS= RR x RS
    assert "RS0 = R x RS0" in strategy_name_list
    assert "RS1 = R x RS1" in strategy_name_list

    # S01R = S01R x RR
    assert "S01R = S01 x RR_0" in strategy_name_list
    assert "S01R = S01 x RR_1" in strategy_name_list
    assert "S01R = S01 x RR_2" in strategy_name_list

    # RS01 = RR x RS01
    assert "RS01 = R x RS01" in strategy_name_list

    for strategy in strategies_vector:
        input_sharding_spec = strategy.get_sharding_spec_by_name("input_1")
        weight_sharding_spec = strategy.get_sharding_spec_by_name("others")
        output_sharding_spec = strategy.get_sharding_spec_by_name("embedding")

        # make sure the sharding matches across different operation data
        assert output_sharding_spec.sharding_sequence[-1] == weight_sharding_spec.sharding_sequence[-1]
        assert input_sharding_spec.sharding_sequence == output_sharding_spec.sharding_sequence[:-1]


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_embedding_module_handler():
    spawn(check_embedding_module_handler, 4)


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_embedding_function_handler():
    spawn(check_embedding_function_handler, 4)


if __name__ == "__main__":
    test_embedding_module_handler()
    test_embedding_function_handler()
