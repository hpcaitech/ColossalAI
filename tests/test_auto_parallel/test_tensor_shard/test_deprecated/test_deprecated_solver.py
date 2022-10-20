from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.deprecated import Solver
from colossalai.auto_parallel.tensor_shard.deprecated.cost_graph import CostGraph
from colossalai.auto_parallel.tensor_shard.deprecated.graph_analysis import GraphAnalyser
from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3)
        self.conv3 = nn.Conv2d(c_out, c_out, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x * 2
        x = self.conv1(x)
        x = self.conv2(x)
        x = x / 2
        x = self.conv3(x)
        x = self.relu(x)
        return x


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_solver():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    model = ConvModel(16, 32)
    input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}

    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %conv1 : [#users=1] = call_module[target=conv1](args = (%mul,), kwargs = {})
    #     %conv2 : [#users=1] = call_module[target=conv2](args = (%conv1,), kwargs = {})
    #     %truediv : [#users=1] = call_function[target=operator.truediv](args = (%conv2, 2), kwargs = {})
    #     %conv3 : [#users=1] = call_module[target=conv3](args = (%truediv,), kwargs = {})
    #     %relu : [#users=1] = call_module[target=relu](args = (%conv3,), kwargs = {})
    #     return relu
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)

    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    graph_analyser = GraphAnalyser(gm)
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    ret = solver.call_solver_serialized_args()

    # [ 0 0 13 13 13 13 13 0]
    strategies_combination_list = ret[0]
    assert solver.leaf_strategies[2][13].name == 'S01R = S01R x RR'


if __name__ == '__main__':
    test_solver()
