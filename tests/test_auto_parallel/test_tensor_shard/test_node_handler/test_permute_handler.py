from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from colossalai.auto_parallel.tensor_shard.node_handler.conv_handler import ConvFunctionHandler
from colossalai.auto_parallel.tensor_shard.node_handler.experimental import PermuteHandler
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


class ConvPermuteModel(nn.Module):

    def __init__(self, permute_dims):
        super().__init__()
        self.permute_dims = permute_dims

    def forward(self, input, other):
        conv_node = nn.functional.conv2d(input, other, bias=None)
        permute_node = torch.permute(conv_node, self.permute_dims)
        return permute_node


class LinearPermuteModel(nn.Module):

    def __init__(self, tgt_shape):
        super().__init__()
        self.tgt_shape = tgt_shape

    def forward(self, input, other):
        linear_node = nn.functional.linear(input, other, bias=None)
        permute_node = torch.permute(linear_node, self.tgt_shape)
        return permute_node


def check_view_handler(rank, permute_dims, model_cls, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = model_cls(permute_dims).cuda()

    if model_cls.__name__ == 'ConvPermuteModel':
        input = torch.rand(8, 8, 66, 66).to('cuda')
        other = torch.rand(16, 8, 3, 3).to('cuda')
        # index of conv node in computation graph
        node_index = 2
        # total number of conv strategies
        strategy_number = 16
    if model_cls.__name__ == 'LinearPermuteModel':
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
    if model_cls.__name__ == 'ConvPermuteModel':
        # graph():
        #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
        #     %other : torch.Tensor [#users=1] = placeholder[target=other]
        #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%input_1, %other), kwargs = {bias: None})
        #     %permute : [#users=1] = call_function[target=torch.permute](args = (%conv2d, (0, 2, 1, 3)), kwargs = {})
        #     return permute
        graph = tracer.trace(model,
                             meta_args={
                                 "input": torch.rand(8, 8, 66, 66).to('meta'),
                                 "other": torch.rand(16, 8, 3, 3).to('meta'),
                             })

    if model_cls.__name__ == 'LinearPermuteModel':
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
    if model_cls.__name__ == 'ConvPermuteModel':

        conv_handler = ConvFunctionHandler(node=previous_mod_node,
                                           device_mesh=device_mesh,
                                           strategies_vector=previous_strategies_vector)
        conv_handler.register_strategy(compute_resharding_cost=False)
        setattr(previous_mod_node, 'strategies_vector', previous_strategies_vector)

    if model_cls.__name__ == 'LinearPermuteModel':
        assert len(previous_strategies_vector) == 0
        linear_handler = LinearFunctionHandler(node=previous_mod_node,
                                               device_mesh=device_mesh,
                                               strategies_vector=previous_strategies_vector)
        linear_handler.register_strategy(compute_resharding_cost=False)
        setattr(previous_mod_node, 'strategies_vector', previous_strategies_vector)

    permute_handler = PermuteHandler(node=view_node, device_mesh=device_mesh, strategies_vector=view_strategies_vector)

    permute_handler.register_strategy(compute_resharding_cost=False)

    # check operation data mapping
    mapping = permute_handler.get_operation_data_mapping()

    for name, op_data in mapping.items():
        op_data: OperationData
        # make sure they have valid values
        assert op_data.data is not None

    if model_cls.__name__ == 'ConvPermuteModel':
        assert mapping['input'].name == "conv2d"
    else:
        assert mapping['input'].name == "linear"
    assert mapping['input'].data.is_meta
    assert mapping['input'].data.shape == torch.Size([8, 16, 64, 64])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([8, 16, 64, 64])

    assert mapping['output'].name == "permute"
    assert mapping['output'].data.is_meta
    assert mapping['output'].data.shape == torch.permute(torch.rand(8, 16, 64, 64), permute_dims).shape
    assert mapping['output'].type == OperationDataType.OUTPUT

    # reshape handler is a following strategy handler, so the number of strategies is equal to the predecessor node.
    assert len(view_strategies_vector) == len(previous_strategies_vector)
    strategy_name_list = [strategy.name for strategy in view_strategies_vector]
    if rank == 0:
        for name in strategy_name_list:
            print(name)
    if model_cls.__name__ == 'ConvPermuteModel':

        if permute_dims == (0, 2, 1, 3):
            assert '[S0, S1, R, R] -> [S0, R, S1, R]_0' in strategy_name_list
            assert '[S1, S0, R, R] -> [S1, R, S0, R]_1' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R]_2' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R]_3' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R]_4' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R]_5' in strategy_name_list
            assert '[R, S1, R, R] -> [R, R, S1, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> [R, R, S0, R]_7' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_8' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_9' in strategy_name_list
            assert '[R, S0, R, R] -> [R, R, S0, R]_10' in strategy_name_list
            assert '[R, S1, R, R] -> [R, R, S1, R]_11' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_12' in strategy_name_list
            assert '[S01, R, R, R] -> [S01, R, R, R]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_14' in strategy_name_list
            assert '[R, S01, R, R] -> [R, R, S01, R]_15' in strategy_name_list

        if permute_dims == (2, 0, 1, 3):
            assert '[S0, S1, R, R] -> [R, S0, S1, R]_0' in strategy_name_list
            assert '[S1, S0, R, R] -> [R, S1, S0, R]_1' in strategy_name_list
            assert '[S0, R, R, R] -> [R, S0, R, R]_2' in strategy_name_list
            assert '[S1, R, R, R] -> [R, S1, R, R]_3' in strategy_name_list
            assert '[S0, R, R, R] -> [R, S0, R, R]_4' in strategy_name_list
            assert '[S1, R, R, R] -> [R, S1, R, R]_5' in strategy_name_list
            assert '[R, S1, R, R] -> [R, R, S1, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> [R, R, S0, R]_7' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_8' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_9' in strategy_name_list
            assert '[R, S0, R, R] -> [R, R, S0, R]_10' in strategy_name_list
            assert '[R, S1, R, R] -> [R, R, S1, R]_11' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_12' in strategy_name_list
            assert '[S01, R, R, R] -> [R, S01, R, R]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_14' in strategy_name_list
            assert '[R, S01, R, R] -> [R, R, S01, R]_15' in strategy_name_list

    if model_cls.__name__ == 'LinearPermuteModel':

        if permute_dims == (0, 2, 1, 3):
            assert '[S0, R, R, S1] -> [S0, R, R, S1]_0' in strategy_name_list
            assert '[R, S0, R, S1] -> [R, R, S0, S1]_1' in strategy_name_list
            assert '[R, R, S0, S1] -> [R, S0, R, S1]_2' in strategy_name_list
            assert '[S1, R, R, S0] -> [S1, R, R, S0]_3' in strategy_name_list
            assert '[R, S1, R, S0] -> [R, R, S1, S0]_4' in strategy_name_list
            assert '[R, R, S1, S0] -> [R, S1, R, S0]_5' in strategy_name_list
            assert '[S0, R, R, R] -> [S0, R, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> [R, R, S0, R]_7' in strategy_name_list
            assert '[R, R, S0, R] -> [R, S0, R, R]_8' in strategy_name_list
            assert '[S1, R, R, R] -> [S1, R, R, R]_9' in strategy_name_list
            assert '[R, S1, R, R] -> [R, R, S1, R]_10' in strategy_name_list
            assert '[R, R, S1, R] -> [R, S1, R, R]_11' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, S1]_12' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, S0]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_15' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, S0]_16' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, S1]_17' in strategy_name_list
            assert '[S01, R, R, R] -> [S01, R, R, R]_18' in strategy_name_list
            assert '[R, S01, R, R] -> [R, R, S01, R]_19' in strategy_name_list
            assert '[R, R, S01, R] -> [R, S01, R, R]_20' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_21' in strategy_name_list
            assert '[R, R, R, S01] -> [R, R, R, S01]_22' in strategy_name_list

        if permute_dims == (2, 0, 1, 3):
            assert '[S0, R, R, S1] -> [R, S0, R, S1]_0' in strategy_name_list
            assert '[R, S0, R, S1] -> [R, R, S0, S1]_1' in strategy_name_list
            assert '[R, R, S0, S1] -> [S0, R, R, S1]_2' in strategy_name_list
            assert '[S1, R, R, S0] -> [R, S1, R, S0]_3' in strategy_name_list
            assert '[R, S1, R, S0] -> [R, R, S1, S0]_4' in strategy_name_list
            assert '[R, R, S1, S0] -> [S1, R, R, S0]_5' in strategy_name_list
            assert '[S0, R, R, R] -> [R, S0, R, R]_6' in strategy_name_list
            assert '[R, S0, R, R] -> [R, R, S0, R]_7' in strategy_name_list
            assert '[R, R, S0, R] -> [S0, R, R, R]_8' in strategy_name_list
            assert '[S1, R, R, R] -> [R, S1, R, R]_9' in strategy_name_list
            assert '[R, S1, R, R] -> [R, R, S1, R]_10' in strategy_name_list
            assert '[R, R, S1, R] -> [S1, R, R, R]_11' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, S1]_12' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, S0]_13' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_14' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_15' in strategy_name_list
            assert '[R, R, R, S0] -> [R, R, R, S0]_16' in strategy_name_list
            assert '[R, R, R, S1] -> [R, R, R, S1]_17' in strategy_name_list
            assert '[S01, R, R, R] -> [R, S01, R, R]_18' in strategy_name_list
            assert '[R, S01, R, R] -> [R, R, S01, R]_19' in strategy_name_list
            assert '[R, R, S01, R] -> [S01, R, R, R]_20' in strategy_name_list
            assert '[R, R, R, R] -> [R, R, R, R]_21' in strategy_name_list
            assert '[R, R, R, S01] -> [R, R, R, S01]_22' in strategy_name_list


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
@parameterize('permute_dims', [(0, 2, 1, 3), (2, 0, 1, 3)])
@parameterize('model_cls', [ConvPermuteModel, LinearPermuteModel])
def test_view_handler(permute_dims, model_cls):
    world_size = 4
    run_func = partial(check_view_handler,
                       permute_dims=permute_dims,
                       model_cls=model_cls,
                       world_size=world_size,
                       port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_view_handler()
