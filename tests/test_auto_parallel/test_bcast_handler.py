import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.auto_parallel.solver.options import SolverOptions
from colossalai.auto_parallel.solver.strategies_constructor import StrategiesConstructor
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.device.device_mesh import DeviceMesh


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 + x2
        x = x1 + 1
        return x


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
    #     %x : torch.Tensor [#users=2] = placeholder[target=x]
    #     %conv1 : [#users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    #     %conv2 : [#users=1] = call_module[target=conv2](args = (%x,), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%conv1, %conv2), kwargs = {})
    #     %add_1 : [#users=1] = call_function[target=operator.add](args = (%add, 1), kwargs = {})
    #     return add_1
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    # [x, conv1, conv2, add, add1, output]
    nodes = [node for node in gm.graph.nodes]
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    strategies_constructor.build_strategies_and_cost()
    strategy_map = strategies_constructor.strategy_map
    # check two tensors element-wise add case
    add_strategies = strategy_map[nodes[3]]
    assert len(add_strategies) == 25

    # check a tensor add with a scalar case
    conv1_strategies = strategy_map[nodes[1]]
    add_1_strategies = strategy_map[nodes[4]]
    add_1_strategies_cover_list = [strategy.input_shardings[0].sharding_sequence for strategy in add_1_strategies]
    for strategy in conv1_strategies:
        assert strategy.output_sharding_spec.sharding_sequence in add_1_strategies_cover_list


if __name__ == '__main__':
    test_conv_handler()
