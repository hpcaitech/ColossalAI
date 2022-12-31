from faulthandler import disable
from functools import partial
from xml.dom import WrongDocumentErr

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from typing_extensions import Self

from colossalai.auto_parallel.tensor_shard.node_handler import LinearFunctionHandler
from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    OperationData,
    OperationDataType,
    ShardingStrategy,
    StrategiesVector,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_node_handler.utils import numerical_test_for_node_strategy


class AddmmModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, m1, m2):
        x = torch.addmm(input, m1, m2, beta=3, alpha=2)
        return x


class AddmmModel_with_param(nn.Module):

    def __init__(self, weight_shape, bias_shape):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(weight_shape))
        self.bias = torch.nn.Parameter(torch.rand(bias_shape))

    def forward(self, m1):
        x = torch.addmm(self.bias, m1, self.weight, beta=3, alpha=2)
        return x


def check_addmm_function_handler(rank, input_shape, model_cls, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    if model_cls == AddmmModel:
        model = AddmmModel().cuda()
    else:
        model = AddmmModel_with_param(weight_shape=(8, 16), bias_shape=input_shape).cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    if model_cls == AddmmModel:
        input = torch.rand(input_shape).cuda()
        m1 = torch.rand(4, 8).cuda()
        m2 = torch.rand(8, 16).cuda()
        # construct input args
        input_args = [input, m1, m2]
        # construct meta arg names
        meta_arg_names = ['input', 'm1', 'm2']
        meta_args_for_tracer = {}
        for meta_arg, input_arg in zip(meta_arg_names, input_args):
            meta_args_for_tracer[meta_arg] = input_arg.to('meta')

        # the index of addmm node in computation graph
        node_index = 4
        # strategy number of linear node
        strategy_number = 14
    else:
        m1 = torch.rand(4, 8).cuda()
        # construct input args
        input_args = [m1]
        # construct meta arg names
        meta_arg_names = ['m1']
        # the index of addmm node in computation graph
        meta_args_for_tracer = {}
        for meta_arg, input_arg in zip(meta_arg_names, input_args):
            meta_args_for_tracer[meta_arg] = input_arg.to('meta')
        node_index = 4
        # strategy number of linear node
        strategy_number = 14

    numerical_test_for_node_strategy(model=model,
                                     device_mesh=device_mesh,
                                     node_index=node_index,
                                     strategy_number=strategy_number,
                                     input_args=input_args,
                                     meta_arg_names=meta_arg_names,
                                     node_type='bias_module')

    tracer = ColoTracer()
    # graph():
    #     %input_1 : torch.Tensor [#users=1] = placeholder[target=input]
    #     %m1 : torch.Tensor [#users=1] = placeholder[target=m1]
    #     %m2 : torch.Tensor [#users=1] = placeholder[target=m2]
    #     %transpose : [#users=1] = call_function[target=torch.transpose](args = (%m2, 0, 1), kwargs = {})
    #     %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%m1, %transpose), kwargs = {})
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%input_1, 3), kwargs = {})
    #     %mul_1 : [#users=1] = call_function[target=operator.mul](args = (2, %linear), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%mul_1, %mul), kwargs = {})
    #     return add
    graph = tracer.trace(model, meta_args=meta_args_for_tracer)
    gm = ColoGraphModule(model, graph)
    # [input_1, m1, m2, addmm, output]
    node_list = list(graph.nodes)
    linear_node = node_list[4]
    strategies_vector = StrategiesVector(linear_node)

    # build handler
    handler = LinearFunctionHandler(node=linear_node, device_mesh=device_mesh, strategies_vector=strategies_vector)

    handler.register_strategy(compute_resharding_cost=False)
    strategy_name_list = [val.name for val in strategies_vector]

    # check operation data mapping
    mapping = handler.get_operation_data_mapping()

    assert mapping['input'].name == "m1"
    assert mapping['input'].data.shape == torch.Size([4, 8])
    assert mapping['input'].type == OperationDataType.ARG
    assert mapping['input'].logical_shape == torch.Size([4, 8])

    assert mapping['other'].name == "transpose"
    assert mapping['other'].data.shape == torch.Size([16, 8])
    if model_cls == AddmmModel:
        assert mapping['other'].type == OperationDataType.ARG
    else:
        assert mapping['other'].type == OperationDataType.PARAM
    assert mapping['other'].logical_shape == torch.Size([8, 16])

    assert mapping['output'].name == "linear"
    assert mapping['output'].data.shape == torch.Size([4, 16])
    assert mapping['output'].type == OperationDataType.OUTPUT

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1_0' in strategy_name_list
    assert 'S1S0 = S1R x RS0_0' in strategy_name_list

    # SR = SS x SR
    assert 'S0R = S0S1 x S1R_0' in strategy_name_list
    assert 'S1R = S1S0 x S0R_0' in strategy_name_list

    # RS = RS x SS
    assert 'RS0 = RS1 x S1S0' in strategy_name_list
    assert 'RS1 = RS0 x S0S1' in strategy_name_list

    # RR = RS x SR
    assert 'RR = RS0 x S0R' in strategy_name_list
    assert 'RR = RS1 x S1R' in strategy_name_list

    # RS= RR x RS
    assert 'RS0 = RR x RS0' in strategy_name_list
    assert 'RS1 = RR x RS1' in strategy_name_list

    # S01R = S01R x RR
    assert 'S01R = S01R x RR_0' in strategy_name_list

    # RR = RS01 x S01R
    assert 'RR = RS01 x S01R' in strategy_name_list

    # RS01 = RR x RS01
    assert 'RS01 = RR x RS01' in strategy_name_list

    # RR = RR x RR
    assert 'RR = RR x RR' in strategy_name_list

    for strategy in strategies_vector:
        strategy: ShardingStrategy
        input_sharding_spec = strategy.get_sharding_spec_by_name('m1')
        weight_sharding_spec = strategy.get_sharding_spec_by_name('transpose')
        output_sharding_spec = strategy.get_sharding_spec_by_name('linear')

        # make sure the sharding matches across different operation data
        assert input_sharding_spec.sharding_sequence[:-1] == output_sharding_spec.sharding_sequence[:-1]
        assert weight_sharding_spec.sharding_sequence[1] == input_sharding_spec.sharding_sequence[1]
        assert weight_sharding_spec.sharding_sequence[0] == output_sharding_spec.sharding_sequence[1]


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@parameterize('input_shape', [(16,), (4, 16)])
@parameterize('model_cls', [AddmmModel, AddmmModel_with_param])
@rerun_if_address_is_in_use()
def test_addmm_handler(input_shape, model_cls):
    world_size = 4
    run_func_function = partial(check_addmm_function_handler,
                                input_shape=input_shape,
                                model_cls=model_cls,
                                world_size=world_size,
                                port=free_port())
    mp.spawn(run_func_function, nprocs=world_size)


if __name__ == '__main__':
    test_addmm_handler()
