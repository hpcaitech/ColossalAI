from cProfile import run

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1 + 1
        x1 = torch.reshape(x1, [1, -1, 64, 1])
        x3 = self.conv2(x1)
        x3 = torch.reshape(x3, [4, 1, 64, -1])
        x = x1 + x3

        return x


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_conv_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

    tracer = ColoTracer()
    model = ConvModel(16, 32)
    input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv1 : [#users=2] = call_module[target=conv1](args = (%x,), kwargs = {})
    #     %add : [#users=0] = call_function[target=operator.add](args = (%conv1, 1), kwargs = {})
    #     %reshape : [#users=2] = call_function[target=torch.reshape](args = (%conv1, [1, -1, 64, 1]), kwargs = {})
    #     %conv2 : [#users=1] = call_module[target=conv2](args = (%reshape,), kwargs = {})
    #     %reshape_1 : [#users=1] = call_function[target=torch.reshape](args = (%conv2, [4, 1, 64, -1]), kwargs = {})
    #     %add_1 : [#users=1] = call_function[target=operator.add](args = (%reshape, %reshape_1), kwargs = {})
    #     return add_1
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    # [x, conv1, add, reshape, conv2, reshape_1, add_1, output]
    nodes = [node for node in gm.graph.nodes]
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    strategies_constructor.build_strategies_and_cost()
    strategy_map = strategies_constructor.strategy_map
    # check a tensor add with a scalar case
    conv1_strategies = strategy_map[nodes[1]]
    add_strategies = strategy_map[nodes[2]]
    add_strategies_cover_list = [strategy.input_shardings[0].sharding_sequence for strategy in add_strategies]
    for strategy in conv1_strategies:
        assert strategy.output_sharding_spec.sharding_sequence in add_strategies_cover_list

    # check two tensors element-wise add case
    add_1_strategies = strategy_map[nodes[6]]
    assert len(add_1_strategies) == 25


if __name__ == '__main__':
    test_conv_handler()
