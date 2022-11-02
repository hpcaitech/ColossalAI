from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.deprecated.op_handler.conv_handler import CONV_STRATEGIES_LIST
from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec


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

    tracer = ColoTracer()
    model = ConvModel(16, 32)
    input_sample = {'x': torch.rand(4, 16, 64, 64).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %conv_weight : [#users=1] = get_attr[target=conv.weight]
    #     %conv_bias : [#users=1] = get_attr[target=conv.bias]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%mul, %conv_weight), kwargs = {groups: 1, dilation: (1, 1), stride: (1, 1), padding: (0, 0)})
    #     %view : [#users=1] = call_method[target=view](args = (%conv_bias, [1, -1, 1, 1]), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%conv2d, %view), kwargs = {})
    #     return add
    graph = tracer.trace(root=model, meta_args=input_sample)
    print(graph)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    assert strategies_constructor.leaf_strategies == []
    assert strategies_constructor.strategy_map == {}
    strategies_constructor.build_strategies_and_cost()

    # check leaf_strategies

    # In fast mode, placeholder node only has replica strategy.
    assert strategies_constructor.leaf_strategies[0][0].name == 'Replica Placeholder'

    # Second node is mul which is a element-wise node, therefore the output sharding spec is same as input sharding spec.
    assert strategies_constructor.leaf_strategies[1][0].name == '[R, R, R, R] -> [R, R, R, R]_0'

    # Third node is conv.
    conv_check_list = deepcopy(CONV_STRATEGIES_LIST)
    for strategy in strategies_constructor.leaf_strategies[4]:
        conv_check_list.remove(strategy.name)
    assert len(conv_check_list) == 0

    # In fast mode, output node only has replica strategy.
    assert strategies_constructor.leaf_strategies[7][0].name == 'Replica Output'

    # check strategy_map

    nodes = [node for node in graph.nodes]
    # In fast mode, placeholder node only has replica strategy.
    x = nodes[0]
    assert strategies_constructor.strategy_map[x][0].name == 'Replica Placeholder'

    # Second node is mul which is a element-wise node, therefore the output sharding spec is same as input sharding spec.
    mul = nodes[1]
    assert strategies_constructor.strategy_map[mul][0].name == '[R, R, R, R] -> [R, R, R, R]_0'

    # fifth node is conv.
    conv = nodes[4]
    conv_check_list = deepcopy(CONV_STRATEGIES_LIST)
    for strategy in strategies_constructor.strategy_map[conv]:
        conv_check_list.remove(strategy.name)
    assert len(conv_check_list) == 0

    # In fast mode, output node only has replica strategy.
    output = nodes[-1]
    assert strategies_constructor.strategy_map[output][0].name == 'Replica Output'


if __name__ == '__main__':
    test_strategies_constructor()
