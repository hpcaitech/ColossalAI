from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.experimental import ViewHandler
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


class ConvViewModel(nn.Module):

    def __init__(self, tgt_shape):
        super().__init__()
        self.tgt_shape = tgt_shape

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other, bias=None)
        reshape_node = conv_node.view(*self.tgt_shape)
        return reshape_node


class LinearViewModel(nn.Module):

    def __init__(self, tgt_shape):
        super().__init__()
        self.tgt_shape = tgt_shape

    def forward(self, input, other):
        linear_node = nn.functional.linear(input, other, bias=None)
        reshape_node = linear_node.view(*self.tgt_shape)
        return reshape_node


def check_view_handler(rank, tgt_shape, model_cls, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = model_cls(tgt_shape).cuda()

    if model_cls.__name__ == 'ConvViewModel':
        input = torch.rand(8, 8, 66, 66).to('cuda')
        other = torch.rand(16, 8, 3, 3).to('cuda')
        # index of conv node in computation graph
        node_index = 2
        # total number of conv strategies
        strategy_number = 16
    if model_cls.__name__ == 'LinearViewModel':
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
    if model_cls.__name__ == 'ConvViewModel':
        # graph():
        #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
        #     %other : torch.Tensor [#users=1] = placeholder[target=other]
        #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {})
        #     %view : [#users=1] = call_method[target=view](args = (%conv2d, 2, -1), kwargs = {})
        #     return view
        graph = tracer.trace(model,
                             meta_args={
                                 "input": torch.rand(8, 8, 66, 66).to('meta'),
                                 "other": torch.rand(16, 8, 3, 3).to('meta'),
                             })

    if model_cls.__name__ == 'LinearViewModel':
        # graph():
        #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
        #     %other : torch.Tensor [#users=1] = placeholder[target=other]
        #     %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%input_1, %other), kwargs = {bias: None})
        #     %view : [#users=1] = call_method[target=view](args = (%linear, 32, 4, 32, 32, 4), kwargs = {})
        #     return view
        graph = tracer.trace(model,
                             meta_args={
                                 "input": torch.rand(8, 16, 64, 32).to('meta'),
                                 "other": torch.rand(64, 32).to('meta'),
                             })

    gm = ColoGraphModule(model, graph)

    previous_mod_node = list(graph.nodes)[2]
    view_node = list(graph.nodes)[3]
    view_strategies_vector = StrategiesVector(view_node)
    previous_strategies_vector = StrategiesVector(previous_mod_node)

    # build handler
    if model_cls.__name__ == 'ConvViewModel':

        conv_handler = ConvFunctionHandler(node=previous_mod_node,
                                           device_mesh=device_mesh,
                                           strategies_vector=previous_strategies_vector)
        conv_handler.register_strategy(compute_resharding_cost=False)
        setattr(previous_mod_node, 'strategies_vector', previous_strategies_vector)

    if model_cls.__name__ == 'LinearViewModel':
        assert len(previous_strategies_vector) == 0
        linear_handler = LinearFunctionHandler(node=previous_mod_node,
                                               device_mesh=device_mesh,
                                               strategies_vector=previous_strategies_vector)
        linear_handler.register_strategy(compute_resharding_cost=False)
        setattr(previous_mod_node, 'strategies_vector', previous_strategies_vector)

    view_handler = ViewHandler(node=view_node, device_mesh=device_mesh, strategies_vector=view_strategies_vector)

    view_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = view_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    if model_cls.__name__ == 'ConvViewModel':
        assert mapping['input'].name == "conv2d"
    else:
        assert mapping['input'].name == "linear"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([8, 16, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([8, 16, 64, 64])

    assert mapping['output'].name == "view"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.Size(tgt_shape)
    assert mapping['output'].type == OperationDataType.OUTPUT

    # reshape handler is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(view_strategies_vector) == len(previous_strategies_vector)
    strategy_name_list = [strategy.name for strategy in view_strategies_vector]

    if model_cls.__name__ == 'ConvViewModel':

        if tgt_shape == (32, 4, 64, 16, 4):
            assert '[S0, S1, R, R] -> FULLY REPLICATED_0' in strategy_name_list
            assert '[S1, S0, R, R] -> FULLY REPLICATED_1' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R, R]_2' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R, R]_3' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R, R]_4' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R, R]_5' in strategy_name_list
            assert '[R, S1, R, R] -> FULLY REPLICATED_6' in strategy_name_list
            assert '[R, S0, R, R] -> FULLY REPLICATED_7' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_8' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_9' in strategy_name_list
            assert '[R, S0, R, R] -> FULLY REPLICATED_10' in strategy_name_list
            assert '[R, S1, R, R] -> FULLY REPLICATED_11' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_12' in strategy_name_list
            assert '[S01, R, R, R] -> [S01, R, R, R, R]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_14' in strategy_name_list
            assert '[R, S01, R, R] -> FULLY REPLICATED_15' in strategy_name_list

        if tgt_shape == (8, 4, 4, 64, 16, 4):
            assert '[S0, S1, R, R] -> [S0, S1, R, R, R, R]_0' in strategy_name_list
            assert '[S1, S0, R, R] -> [S1, S0, R, R, R, R]_1' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R, R, R]_2' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R, R, R]_3' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R, R, R]_4' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R, R, R]_5' in strategy_name_list
            assert '[R, S1, R, R] -> [R, S1, R, R, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> [R, S0, R, R, R, R]_7' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_8' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_9' in strategy_name_list
            assert '[R, S0, R, R] -> [R, S0, R, R, R, R]_10' in strategy_name_list
            assert '[R, S1, R, R] -> [R, S1, R, R, R, R]_11' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_12' in strategy_name_list
            assert '[S01, R, R, R] -> [S01, R, R, R, R, R]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_14' in strategy_name_list
            assert '[R, S01, R, R] -> [R, S01, R, R, R, R]_15' in strategy_name_list

    if model_cls.__name__ == 'LinearViewModel':

        if tgt_shape == (32, 4, 64, 16, 4):
            assert '[S0, R, R, S1] -> [S0, R, R, S1, R]_0' in strategy_name_list
            assert '[R, S0, R, S1] -> FULLY REPLICATED_1' in strategy_name_list
            assert '[R, R, S0, S1] -> [R, R, S0, S1, R]_2' in strategy_name_list
            assert '[S1, R, R, S0] -> [S1, R, R, S0, R]_3' in strategy_name_list
            assert '[R, S1, R, S0] -> FULLY REPLICATED_4' in strategy_name_list
            assert '[R, R, S1, S0] -> [R, R, S1, S0, R]_5' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> FULLY REPLICATED_7' in strategy_name_list
            assert '[R, R, S0, R] -> [R, R, S0, R, R]_8' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R, R]_9' in strategy_name_list
            assert '[R, S1, R, R] -> FULLY REPLICATED_10' in strategy_name_list
            assert '[R, R, S1, R] -> [R, R, S1, R, R]_11' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, S1, R]_12' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, S0, R]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_15' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, S0, R]_16' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, S1, R]_17' in strategy_name_list
            assert '[S01, R, R, R] -> [S01, R, R, R, R]_18' in strategy_name_list
            assert '[R, S01, R, R] -> FULLY REPLICATED_19' in strategy_name_list
            assert '[R, R, S01, R] -> [R, R, S01, R, R]_20' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R]_21' in strategy_name_list
            assert '[R, R, R, S01] -> [R, R, R, S01, R]_22' in strategy_name_list

        if tgt_shape == (8, 4, 4, 64, 16, 4):
            assert '[S0, R, R, S1] -> [S0, R, R, R, S1, R]_0' in strategy_name_list
            assert '[R, S0, R, S1] -> [R, S0, R, R, S1, R]_1' in strategy_name_list
            assert '[R, R, S0, S1] -> [R, R, R, S0, S1, R]_2' in strategy_name_list
            assert '[S1, R, R, S0] -> [S1, R, R, R, S0, R]_3' in strategy_name_list
            assert '[R, S1, R, S0] -> [R, S1, R, R, S0, R]_4' in strategy_name_list
            assert '[R, R, S1, S0] -> [R, R, R, S1, S0, R]_5' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> [R, S0, R, R, R, R]_7' in strategy_name_list
            assert '[R, R, S0, R] -> [R, R, R, S0, R, R]_8' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R, R, R]_9' in strategy_name_list
            assert '[R, S1, R, R] -> [R, S1, R, R, R, R]_10' in strategy_name_list
            assert '[R, R, S1, R] -> [R, R, R, S1, R, R]_11' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, R, S1, R]_12' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, R, S0, R]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_15' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, R, S0, R]_16' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, R, S1, R]_17' in strategy_name_list
            assert '[S01, R, R, R] -> [S01, R, R, R, R, R]_18' in strategy_name_list
            assert '[R, S01, R, R] -> [R, S01, R, R, R, R]_19' in strategy_name_list
            assert '[R, R, S01, R] -> [R, R, R, S01, R, R]_20' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R, R, R]_21' in strategy_name_list
            assert '[R, R, R, S01] -> [R, R, R, R, S01, R]_22' in strategy_name_list


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
@parameterize('tgt_shape', [(32, 4, 64, 16, 4), (8, 4, 4, 64, 16, 4)])
@parameterize('model_cls', [ConvViewModel, LinearViewModel])
def test_view_handler(tgt_shape, model_cls):
    world_size = 4
    run_func = partial(check_view_handler,
                       tgt_shape=tgt_shape,
                       model_cls=model_cls,
                       world_size=world_size,
                       port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_view_handler()
