import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from colossalai.auto_parallel.solver.conv_handler import ConvHandler, CONV_STRATEGIES_LIST
from colossalai.auto_parallel.solver.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.device.device_mesh import DeviceMesh
from colossalai.auto_parallel.solver.strategies_constructor import StrategiesConstructor
from colossalai.auto_parallel.solver.cost_graph import CostGraph
from copy import deepcopy


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x * 2
        x = self.conv1(x)
        x = x / 2
        x = self.relu(x)
        return x


def test_cost_graph():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((4, 16, 64, 64))
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    model = ConvModel(16, 32)
    input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}

    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %conv1 : [#users=1] = call_module[target=conv1](args = (%mul,), kwargs = {})
    #     %truediv : [#users=1] = call_function[target=operator.truediv](args = (%conv1, 2), kwargs = {})
    #     %relu : [#users=1] = call_module[target=relu](args = (%truediv,), kwargs = {})
    #     return relu
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    solver_options = {'fast_mode': True}
    strategies_constructor = StrategiesConstructor(graph, device_mesh, shape_consistency_manager, solver_options)
    strategies_constructor.build_strategies_and_cost()

    # (x, mul): {(0, 0): 0}
    # (mul, conv1): {(0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 0, (0, 6): 0, (0, 7): 0, (0, 8): 0, (0, 9): 0, (0, 10): 0, (0, 11): 0, (0, 12): 0, (0, 13): 0, (0, 14): 0}
    # (conv1, truediv): {(0, 0): 0, (1, 0): inf, (2, 0): 0, (3, 0): inf, (4, 0): 0, (5, 0): inf, (6, 0): inf, (7, 0): 0, (8, 0): inf, (9, 0): 0, (10, 0): 0, (11, 0): 0, (12, 0): 0, (13, 0): inf, (14, 0): inf, (0, 1): inf, (1, 1): 0, (2, 1): inf, (3, 1): 0, (4, 1): inf, (5, 1): 0, (6, 1): 0, (7, 1): inf, (8, 1): 0, (9, 1): inf, (10, 1): 0, (11, 1): 0, (12, 1): 0, (13, 1): inf, (14, 1): inf, (0, 2): inf, (1, 2): inf, (2, 2): 0, (3, 2): inf, (4, 2): inf, (5, 2): inf, (6, 2): inf, (7, 2): inf, (8, 2): inf, (9, 2): inf, (10, 2): 0, (11, 2): 0, (12, 2): 0, (13, 2): inf, (14, 2): inf, (0, 3): inf, (1, 3): inf, (2, 3): inf, (3, 3): 0, (4, 3): inf, (5, 3): inf, (6, 3): inf, (7, 3): inf, (8, 3): inf, (9, 3): inf, (10, 3): 0, (11, 3): 0, (12, 3): 0, (13, 3): inf, (14, 3): inf, (0, 4): inf, (1, 4): inf, (2, 4): inf, (3, 4): inf, (4, 4): inf, (5, 4): inf, (6, 4): 0, (7, 4): inf, (8, 4): 0, (9, 4): inf, (10, 4): 0, (11, 4): 0, (12, 4): 0, (13, 4): inf, (14, 4): inf, (0, 5): inf, (1, 5): inf, (2, 5): inf, (3, 5): inf, (4, 5): inf, (5, 5): inf, (6, 5): inf, (7, 5): 0, (8, 5): inf, (9, 5): 0, (10, 5): 0, (11, 5): 0, (12, 5): 0, (13, 5): inf, (14, 5): inf, (0, 6): inf, (1, 6): inf, (2, 6): inf, (3, 6): inf, (4, 6): inf, (5, 6): inf, (6, 6): inf, (7, 6): inf, (8, 6): inf, (9, 6): inf, (10, 6): 0, (11, 6): 0, (12, 6): 0, (13, 6): inf, (14, 6): inf, (0, 7): inf, (1, 7): inf, (2, 7): 0, (3, 7): inf, (4, 7): inf, (5, 7): inf, (6, 7): inf, (7, 7): inf, (8, 7): inf, (9, 7): inf, (10, 7): 0, (11, 7): 0, (12, 7): 0, (13, 7): 0, (14, 7): inf, (0, 8): inf, (1, 8): inf, (2, 8): inf, (3, 8): inf, (4, 8): inf, (5, 8): inf, (6, 8): 0, (7, 8): inf, (8, 8): 0, (9, 8): inf, (10, 8): 0, (11, 8): 0, (12, 8): 0, (13, 8): inf, (14, 8): 0}
    # (truediv, relu): {(0, 0): 0, (1, 0): inf, (2, 0): 0, (3, 0): inf, (4, 0): inf, (5, 0): 0, (6, 0): 0, (7, 0): inf, (8, 0): inf, (0, 1): inf, (1, 1): 0, (2, 1): inf, (3, 1): 0, (4, 1): 0, (5, 1): inf, (6, 1): 0, (7, 1): inf, (8, 1): inf, (0, 2): inf, (1, 2): inf, (2, 2): 0, (3, 2): inf, (4, 2): inf, (5, 2): inf, (6, 2): 0, (7, 2): inf, (8, 2): inf, (0, 3): inf, (1, 3): inf, (2, 3): inf, (3, 3): 0, (4, 3): inf, (5, 3): inf, (6, 3): 0, (7, 3): inf, (8, 3): inf, (0, 4): inf, (1, 4): inf, (2, 4): inf, (3, 4): inf, (4, 4): 0, (5, 4): inf, (6, 4): 0, (7, 4): inf, (8, 4): inf, (0, 5): inf, (1, 5): inf, (2, 5): inf, (3, 5): inf, (4, 5): inf, (5, 5): 0, (6, 5): 0, (7, 5): inf, (8, 5): inf, (0, 6): inf, (1, 6): inf, (2, 6): inf, (3, 6): inf, (4, 6): inf, (5, 6): inf, (6, 6): 0, (7, 6): inf, (8, 6): inf, (0, 7): inf, (1, 7): inf, (2, 7): 0, (3, 7): inf, (4, 7): inf, (5, 7): inf, (6, 7): 0, (7, 7): 0, (8, 7): inf, (0, 8): inf, (1, 8): inf, (2, 8): inf, (3, 8): inf, (4, 8): 0, (5, 8): inf, (6, 8): 0, (7, 8): inf, (8, 8): 0}
    # (relu, output): {(0, 0): 246019.30000000002, (1, 0): 246019.30000000002, (2, 0): 123009.1, (3, 0): 123009.1, (4, 0): 123009.1, (5, 0): 123009.1, (6, 0): 0, (7, 0): 246019.30000000002, (8, 0): 246019.30000000002}
    cost_graph = CostGraph(strategies_constructor.leaf_strategies)

    # construct all node pairs
    all_node_pairs = []

    for node in graph.nodes:
        if node.op == 'output':
            continue
        all_node_pairs.append((node, node.next))

    for node_pair in all_node_pairs:
        assert node_pair in cost_graph.edge_costs

    # construct merged node pairs
    merged_node_pairs = []
    node_list = list(graph.nodes)

    # add (x, conv) and (conv, output) into check node pairs
    merged_node_pairs.append((node_list[0], node_list[2]))
    merged_node_pairs.append((node_list[2], node_list[-1]))
    # (x, conv1): {(0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0, (0, 4): 0, (0, 5): 0, (0, 6): 0, (0, 7): 0, (0, 8): 0, (0, 9): 0, (0, 10): 0, (0, 11): 0, (0, 12): 0, (0, 13): 0, (0, 14): 0}
    # (conv1, output): {(0, 0): inf, (1, 0): inf, (2, 0): inf, (3, 0): inf, (4, 0): inf, (5, 0): inf, (6, 0): inf, (7, 0): inf, (8, 0): inf, (9, 0): inf, (10, 0): 0, (11, 0): 0, (12, 0): 0, (13, 0): inf, (14, 0): inf}
    cost_graph.simplify_graph()
    for node_pair in all_node_pairs:
        if node_pair in merged_node_pairs:
            assert node_pair in cost_graph.edge_costs
        else:
            assert node_pair not in cost_graph.edge_costs


if __name__ == '__main__':
    test_cost_graph()
