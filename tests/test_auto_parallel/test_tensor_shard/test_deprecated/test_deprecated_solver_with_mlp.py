import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.device.device_mesh import DeviceMesh
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.auto_parallel.tensor_shard.deprecated.cost_graph import CostGraph
from copy import deepcopy
from colossalai.auto_parallel.tensor_shard.deprecated import Solver
from torchvision.models import resnet34, resnet50
from colossalai.auto_parallel.tensor_shard.deprecated.constants import *
from colossalai.auto_parallel.tensor_shard.deprecated.graph_analysis import GraphAnalyser
from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class MLP(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim * 4)
        self.linear2 = torch.nn.Linear(dim * 4, dim)
        self.dropout = torch.nn.Dropout(0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_cost_graph():
    physical_mesh_id = torch.arange(0, 8)
    mesh_shape = (2, 4)
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    model = MLP(32)

    input_sample = {'x': torch.rand(16, 32).to('meta')}

    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %linear1 : [#users=1] = call_module[target=linear1](args = (%x,), kwargs = {})
    #     %dropout : [#users=1] = call_module[target=dropout](args = (%linear1,), kwargs = {})
    #     %relu : [#users=1] = call_module[target=relu](args = (%dropout,), kwargs = {})
    #     %linear2 : [#users=1] = call_module[target=linear2](args = (%relu,), kwargs = {})
    #     return linear2
    graph = tracer.trace(root=model, meta_args=input_sample)

    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    # # megatron mode if no memory constraints
    # solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    # all sharding on out feature dim if memory budget is not sufficient for megatron mode
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser, memory_budget=5500.0)

    ret = solver.call_solver_serialized_args()
    strategies_list = list(ret[0])
    computation_cost = 0
    communication_cost = 0
    memory_cost = 0
    for index, node in enumerate(graph.nodes):
        print(node.name, node.strategies_vector[strategies_list[index]].name)
        computation_cost += node.strategies_vector[strategies_list[index]].compute_cost
        communication_cost += node.strategies_vector[strategies_list[index]].communication_cost
        node_memory_cost = node.strategies_vector[strategies_list[index]].memory_cost
        if isinstance(node_memory_cost, tuple):
            node_memory_cost = node_memory_cost[0]
        memory_cost += node_memory_cost

    print(f'computation cost is {computation_cost}')
    print(f'communication cost is {communication_cost}')
    print(f'memory cost is {memory_cost}')


if __name__ == '__main__':
    test_cost_graph()
