from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.default_reshape_handler import DefaultReshapeHandler
from colossalai.auto_parallel.tensor_shard.node_handler.getitem_handler import GetItemHandler
from colossalai.auto_parallel.tensor_shard.node_handler.linear_handler import LinearFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.placeholder_handler import PlaceholderHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.fx.tracer.meta_patch.patched_module import linear
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


class GetItemFromTensorModel(nn.Module):

    def __init__(self, getitem_index):
        super().__init__()
        self.getitem_index = getitem_index

    def forward(self, input, other):
        linear_node = nn.functional.linear(input, other, bias=None)
        x = linear_node[self.getitem_index]
        return x


def check_getitem_from_tensor_handler(rank, getitem_index, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    model = GetItemFromTensorModel(getitem_index=getitem_index)

    input = torch.rand(8, 16, 64, 32).to('cuda')
    other = torch.rand(64, 32).to('cuda')
    # index of linear node in computation graph
    node_index = 2
    # total number of linear strategies
    strategy_number = 23

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    numerical_test_for_node_strategy(model=model,
                                     device_mesh=device_mesh,
                                     node_index=node_index,
                                     strategy_number=strategy_number,
                                     input_args=[input, other],
                                     meta_arg_names=['input', 'other'],
                                     node_type='following')

    tracer = ColoTracer()

    graph = tracer.trace(model,
                         meta_args={
                             "input": torch.rand(8, 16, 64, 32).to('meta'),
                             "other": torch.rand(64, 32).to('meta'),
                         })

    gm = ColoGraphModule(model, graph)
    linear_mod_node = list(graph.nodes)[2]
    getitem_mod_node = list(graph.nodes)[3]
    getitem_strategies_vector = StrategiesVector(getitem_mod_node)
    linear_strategies_vector = StrategiesVector(linear_mod_node)

    # build handler
    linear_handler = LinearFunctionHandler(node=linear_mod_node,
                                           device_mesh=device_mesh,
                                           strategies_vector=linear_strategies_vector)
    linear_handler.register_strategy(compute_resharding_cost=False)
    setattr(linear_mod_node, 'strategies_vector', linear_strategies_vector)
    getitem_handler = GetItemHandler(node=getitem_mod_node,
                                     device_mesh=device_mesh,
                                     strategies_vector=getitem_strategies_vector)

    getitem_handler.register_strategy(compute_resharding_cost=False)
    # check operation data mapping
    mapping = getitem_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    # getitem is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(getitem_strategies_vector) == len(linear_strategies_vector)


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
# @parameterize('getitem_index', [slice(0, 2), (slice(None), slice(None))])
@parameterize('getitem_index', [1, (1, 4), slice(0, 2), (slice(None), slice(None))])
def test_getitem_from_tensor_handler(getitem_index):
    world_size = 4
    run_func = partial(check_getitem_from_tensor_handler,
                       getitem_index=getitem_index,
                       world_size=world_size,
                       port=free_port())
    mp.spawn(run_func, nprocs=world_size)


class GetItemFromTupleModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        split_node = torch.split(input, 2, 0)
        x = split_node[1]
        return x


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_getitem_from_tuple_handler():
    model = GetItemFromTupleModel()
    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %split : [#users=1] = call_function[target=torch.functional.split](args = (%conv2d, 2), kwargs = {dim: 0})
    #     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%split, 1), kwargs = {})
    #     return getitem
    graph = tracer.trace(model, meta_args={
        "input": torch.rand(4, 4, 64, 64).to('meta'),
    })
    gm = ColoGraphModule(model, graph)
    physical_mesh_id = torch.arange(0, 4)

    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    input_node = list(graph.nodes)[0]
    split_node = list(graph.nodes)[1]
    getitem_node = list(graph.nodes)[2]
    input_strategies_vector = StrategiesVector(input_node)
    getitem_strategies_vector = StrategiesVector(getitem_node)
    split_strategies_vector = StrategiesVector(split_node)

    # build handler
    input_handler = PlaceholderHandler(
        node=input_node,
        device_mesh=device_mesh,
        strategies_vector=input_strategies_vector,
        placeholder_option='replicated',
    )
    input_handler.register_strategy(compute_resharding_cost=False)
    setattr(input_node, 'strategies_vector', input_strategies_vector)
    split_handler = DefaultReshapeHandler(node=split_node,
                                          device_mesh=device_mesh,
                                          strategies_vector=split_strategies_vector)
    split_handler.register_strategy(compute_resharding_cost=False)
    setattr(split_node, 'strategies_vector', split_strategies_vector)
    getitem_handler = GetItemHandler(node=getitem_node,
                                     device_mesh=device_mesh,
                                     strategies_vector=getitem_strategies_vector)
    getitem_handler.register_strategy(compute_resharding_cost=False)
    setattr(getitem_node, 'strategies_vector', getitem_strategies_vector)

    # check operation data mapping
    mapping = getitem_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    assert mapping['input'].name == "split"
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == (torch.Size([2, 4, 64, 64]), torch.Size([2, 4, 64, 64]))

    assert mapping['index'].name == "index"
    assert isinstance(mapping['index'].data, int)
    assert mapping['index'].type == OperationDataType.ARG

    assert mapping['output'].name == "getitem"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size([2, 4, 64, 64])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # getitem is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(getitem_strategies_vector) == len(split_strategies_vector)


if __name__ == '__main__':
    test_getitem_from_tensor_handler()
    test_getitem_from_tuple_handler()
