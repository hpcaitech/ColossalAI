from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler import SplitHandler
from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.linear_handler import LinearFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationData, OperationDataType, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


class ConvSplitModel(nn.Module):

    def __init__(self, split_size, split_dim):
        super().__init__()
        self.split_size = split_size
        self.split_dim = split_dim

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other, bias=None)
        split_node = conv_node.split(self.split_size, dim=self.split_dim)
        return split_node


class LinearSplitModel(nn.Module):

    def __init__(self, split_size, split_dim):
        super().__init__()
        self.split_size = split_size
        self.split_dim = split_dim

    def forward(self, input, other):
        linear_node = nn.functional.linear(input, other, bias=None)
        split_node = linear_node.split(self.split_size, dim=self.split_dim)
        return split_node


def check_split_handler(rank, split_size, split_dim, model_cls, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = model_cls(split_size=split_size, split_dim=split_dim).cuda()

    if model_cls.__name__ == 'ConvSplitModel':
        input = torch.rand(8, 8, 66, 66).to('cuda')
        other = torch.rand(16, 8, 3, 3).to('cuda')
        # index of conv node in computation graph
        node_index = 2
        # total number of conv strategies
        strategy_number = 16
    if model_cls.__name__ == 'LinearSplitModel':
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
    if model_cls.__name__ == 'ConvSplitModel':
        # graph():
        #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
        #     %other : torch.Tensor [#users=1] = placeholder[target=other]
        #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {})
        #     %split : [#users=1] = call_method[target=split](args = (%conv2d,), kwargs = {})
        #     return split
        graph = tracer.trace(model,
                             meta_args={
                                 "input": torch.rand(8, 8, 66, 66).to('meta'),
                                 "other": torch.rand(16, 8, 3, 3).to('meta'),
                             })

    if model_cls.__name__ == 'LinearSplitModel':
        # graph():
        #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
        #     %other : torch.Tensor [#users=1] = placeholder[target=other]
        #     %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%input_1, %other), kwargs = {bias: None})
        #     %split : [#users=1] = call_method[target=split](args = (%linear,), kwargs = {})
        #     return split
        graph = tracer.trace(model,
                             meta_args={
                                 "input": torch.rand(8, 16, 64, 32).to('meta'),
                                 "other": torch.rand(64, 32).to('meta'),
                             })

    gm = ColoGraphModule(model, graph)

    previous_mod_node = list(graph.nodes)[2]
    split_node = list(graph.nodes)[3]
    split_strategies_vector = StrategiesVector(split_node)
    previous_strategies_vector = StrategiesVector(previous_mod_node)

    # build handler
    if model_cls.__name__ == 'ConvSplitModel':

        conv_handler = ConvFunctionHandler(node=previous_mod_node,
                                           device_mesh=device_mesh,
                                           strategies_vector=previous_strategies_vector)
        conv_handler.register_strategy(compute_resharding_cost=False)
        setattr(previous_mod_node, 'strategies_vector', previous_strategies_vector)

    if model_cls.__name__ == 'LinearSplitModel':
        assert len(previous_strategies_vector) == 0
        linear_handler = LinearFunctionHandler(node=previous_mod_node,
                                               device_mesh=device_mesh,
                                               strategies_vector=previous_strategies_vector)
        linear_handler.register_strategy(compute_resharding_cost=False)
        setattr(previous_mod_node, 'strategies_vector', previous_strategies_vector)

    split_handler = SplitHandler(node=split_node, device_mesh=device_mesh, strategies_vector=split_strategies_vector)

    split_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = split_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    if model_cls.__name__ == 'ConvSplitModel':
        assert mapping['input'].name == "conv2d"
    else:
        assert mapping['input'].name == "linear"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([8, 16, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([8, 16, 64, 64])

    assert mapping['output'].name == "split"
    split_items = torch.empty([8, 16, 64, 64]).split(split_size, split_dim)
    assert mapping['output'].logical_shape == tuple([item.shape for item in split_items])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # reshape handler is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(split_strategies_vector) == len(previous_strategies_vector)
    strategy_name_list = [strategy.name for strategy in split_strategies_vector]

    if model_cls.__name__ == 'ConvSplitModel':

        if split_dim == 0:
            assert '[R, S1, R, R]_0' in strategy_name_list
            assert '[R, S0, R, R]_1' in strategy_name_list
            assert '[R, R, R, R]_2' in strategy_name_list
            assert '[R, R, R, R]_3' in strategy_name_list
            assert '[R, R, R, R]_4' in strategy_name_list
            assert '[R, R, R, R]_5' in strategy_name_list
            assert '[R, S1, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R]_7' in strategy_name_list
            assert '[R, R, R, R]_8' in strategy_name_list
            assert '[R, R, R, R]_9' in strategy_name_list
            assert '[R, S0, R, R]_10' in strategy_name_list
            assert '[R, S1, R, R]_11' in strategy_name_list
            assert '[R, R, R, R]_12' in strategy_name_list
            assert '[R, R, R, R]_13' in strategy_name_list
            assert '[R, R, R, R]_14' in strategy_name_list
            assert '[R, S01, R, R]_15' in strategy_name_list

        if split_dim == 1:
            assert '[S0, R, R, R]_0' in strategy_name_list
            assert '[S1, R, R, R]_1' in strategy_name_list
            assert '[S0, R, R, R]_2' in strategy_name_list
            assert '[S1, R, R, R]_3' in strategy_name_list
            assert '[S0, R, R, R]_4' in strategy_name_list
            assert '[S1, R, R, R]_5' in strategy_name_list
            assert '[R, R, R, R]_6' in strategy_name_list
            assert '[R, R, R, R]_7' in strategy_name_list
            assert '[R, R, R, R]_8' in strategy_name_list
            assert '[R, R, R, R]_9' in strategy_name_list
            assert '[R, R, R, R]_10' in strategy_name_list
            assert '[R, R, R, R]_11' in strategy_name_list
            assert '[R, R, R, R]_12' in strategy_name_list
            assert '[S01, R, R, R]_13' in strategy_name_list
            assert '[R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R]_15' in strategy_name_list

    if model_cls.__name__ == 'LinearSplitModel':

        if split_dim == 0:
            assert '[R, R, R, S1]_0' in strategy_name_list
            assert '[R, S0, R, S1]_1' in strategy_name_list
            assert '[R, R, S0, S1]_2' in strategy_name_list
            assert '[R, R, R, S0]_3' in strategy_name_list
            assert '[R, S1, R, S0]_4' in strategy_name_list
            assert '[R, R, S1, S0]_5' in strategy_name_list
            assert '[R, R, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R]_7' in strategy_name_list
            assert '[R, R, S0, R]_8' in strategy_name_list
            assert '[R, R, R, R]_9' in strategy_name_list
            assert '[R, S1, R, R]_10' in strategy_name_list
            assert '[R, R, S1, R]_11' in strategy_name_list
            assert '[R, R, R, S1]_12' in strategy_name_list
            assert '[R, R, R, S0]_13' in strategy_name_list
            assert '[R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R]_15' in strategy_name_list
            assert '[R, R, R, S0]_16' in strategy_name_list
            assert '[R, R, R, S1]_17' in strategy_name_list
            assert '[R, R, R, R]_18' in strategy_name_list
            assert '[R, S01, R, R]_19' in strategy_name_list
            assert '[R, R, S01, R]_20' in strategy_name_list
            assert '[R, R, R, R]_21' in strategy_name_list
            assert '[R, R, R, S01]_22' in strategy_name_list

        if split_dim == 1:
            assert '[S0, R, R, S1]_0' in strategy_name_list
            assert '[R, R, R, S1]_1' in strategy_name_list
            assert '[R, R, S0, S1]_2' in strategy_name_list
            assert '[S1, R, R, S0]_3' in strategy_name_list
            assert '[R, R, R, S0]_4' in strategy_name_list
            assert '[R, R, S1, S0]_5' in strategy_name_list
            assert '[S0, R, R, R]_6' in strategy_name_list
            assert '[R, R, R, R]_7' in strategy_name_list
            assert '[R, R, S0, R]_8' in strategy_name_list
            assert '[S1, R, R, R]_9' in strategy_name_list
            assert '[R, R, R, R]_10' in strategy_name_list
            assert '[R, R, S1, R]_11' in strategy_name_list
            assert '[R, R, R, S1]_12' in strategy_name_list
            assert '[R, R, R, S0]_13' in strategy_name_list
            assert '[R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R]_15' in strategy_name_list
            assert '[R, R, R, S0]_16' in strategy_name_list
            assert '[R, R, R, S1]_17' in strategy_name_list
            assert '[S01, R, R, R]_18' in strategy_name_list
            assert '[R, R, R, R]_19' in strategy_name_list
            assert '[R, R, S01, R]_20' in strategy_name_list
            assert '[R, R, R, R]_21' in strategy_name_list
            assert '[R, R, R, S01]_22' in strategy_name_list


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
@parameterize('split_size', [2])
@parameterize('split_dim', [0, 1, 2])
@parameterize('model_cls', [ConvSplitModel, LinearSplitModel])
def test_split_handler(split_size, split_dim, model_cls):
    world_size = 4
    run_func = partial(check_split_handler,
                       split_size=split_size,
                       split_dim=split_dim,
                       model_cls=model_cls,
                       world_size=world_size,
                       port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_split_handler()
