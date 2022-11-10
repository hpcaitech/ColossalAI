import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.deprecated.op_handler.dot_handler import DotHandler
from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec


class LinearModel(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x * 2
        x = self.linear(x)
        return x


@pytest.mark.skip('F.linear is not supported in deprecated handler')
def test_dot_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((4, 8))

    tracer = ColoTracer()
    model = LinearModel(8, 16)
    input_sample = {'x': torch.rand(4, 8).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %linear_weight : [#users=1] = get_attr[target=linear.weight]
    #     %linear_bias : [#users=1] = get_attr[target=linear.bias]
    #     %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%mul, %linear_weight), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%linear, %linear_bias), kwargs = {})
    #     return add
    graph = tracer.trace(root=model, meta_args=input_sample)

    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    strategies_constructor.build_strategies_and_cost()
    linear_node = list(graph.nodes)[4]

    # ['S0S1 = S0R x RS1', 'S1S0 = S1R x RS0', 'S0R = S0S1 x S1R', 'S1R = S1S0 x S0R', 'RS1 = RS0 x S0S1', 'RS0 = RS1 x S1S0', 'RS0 = RR x RS0', 'RS1 = RR x RS1', 'RR = RR x RR']
    strategy_name_list = [strategy.name for strategy in linear_node.strategies_vector]

    # SS = SR x RS
    assert 'S0S1 = S0R x RS1' in strategy_name_list
    assert 'S1S0 = S1R x RS0' in strategy_name_list

    # SR = SS x SR
    assert 'S0R = S0S1 x S1R' in strategy_name_list
    assert 'S1R = S1S0 x S0R' in strategy_name_list

    # RS = RS x SS
    assert 'RS0 = RS1 x S1S0' in strategy_name_list
    assert 'RS1 = RS0 x S0S1' in strategy_name_list

    # RR = RS x SR
    assert 'RR = RS0 x S0R' in strategy_name_list
    assert 'RR = RS1 x S1R' in strategy_name_list

    # RS= RR x RS
    assert 'RS0 = RR x RS0' in strategy_name_list
    assert 'RS1 = RR x RS1' in strategy_name_list


if __name__ == '__main__':
    test_dot_handler()
