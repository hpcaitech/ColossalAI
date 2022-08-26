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
from copy import deepcopy


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3)

    def forward(self, x):
        x = x * 2
        x = self.conv(x)
        return x


def test_strategies_constructor():
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
    #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
    #     return conv
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    solver_options = {'fast_mode': True}
    strategies_constructor = StrategiesConstructor(graph, device_mesh, shape_consistency_manager, solver_options)

    assert strategies_constructor.leaf_strategies == []
    assert strategies_constructor.strategy_map == {}
    strategies_constructor.build_strategies_and_cost()

    # check leaf_strategies

    # In fast mode, placeholder node only has replica strategy.
    assert strategies_constructor.leaf_strategies[0][0].name == 'Replica Placeholder'

    # Second node is mul which is a element-wise node, therefore the output sharding spec is same as input sharding spec.
    assert strategies_constructor.leaf_strategies[1][0].name == '[R, R, R, R] -> [R, R, R, R]'

    # Third node is conv.
    conv_check_list = deepcopy(CONV_STRATEGIES_LIST)
    for strategy in strategies_constructor.leaf_strategies[2]:
        conv_check_list.remove(strategy.name)
    assert len(conv_check_list) == 0

    # In fast mode, output node only has replica strategy.
    assert strategies_constructor.leaf_strategies[3][0].name == 'Replica Output'

    # check strategy_map

    nodes = [node for node in graph.nodes]
    # In fast mode, placeholder node only has replica strategy.
    x = nodes[0]
    assert strategies_constructor.strategy_map[x][0].name == 'Replica Placeholder'

    # Second node is mul which is a element-wise node, therefore the output sharding spec is same as input sharding spec.
    mul = nodes[1]
    assert strategies_constructor.strategy_map[mul][0].name == '[R, R, R, R] -> [R, R, R, R]'

    # Third node is conv.
    conv = nodes[2]
    conv_check_list = deepcopy(CONV_STRATEGIES_LIST)
    for strategy in strategies_constructor.strategy_map[conv]:
        conv_check_list.remove(strategy.name)
    assert len(conv_check_list) == 0

    # In fast mode, output node only has replica strategy.
    output = nodes[3]
    assert strategies_constructor.strategy_map[output][0].name == 'Replica Output'


if __name__ == '__main__':
    test_strategies_constructor()
