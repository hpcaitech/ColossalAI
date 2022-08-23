import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from colossalai.auto_parallel.solver.dot_handler import DotHandler
from colossalai.auto_parallel.solver.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.device.device_mesh import DeviceMesh


class LinearModel(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x * 2
        x = self.linear(x)
        return x


def test_dot_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((4, 8))
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()
    model = LinearModel(8, 16)
    input_sample = {'x': torch.rand(4, 8).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
    #     return conv
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    # [x, mul, linear, output]
    nodes = [node for node in gm.graph.nodes]

    # find the sharding strategies for the input node of the conv node
    # strategies_for_input = [[R, R, R, R], [R, S0, R, R], [R, S1, R, R], [S0, R, R, R], [S0, S1, R, R], [S1, R, R, R], [S1, S0, R, R]]
    strategies_vector_for_input = StrategiesVector(node=nodes[1])
    sharding_option = (None, 0, 1)
    for first_sharding_index in sharding_option:
        for second_sharding_index in sharding_option:
            if first_sharding_index is not None and second_sharding_index == first_sharding_index:
                continue
            if first_sharding_index is None:
                first_dim_spec = _DimSpec([])
            else:
                first_dim_spec = _DimSpec([first_sharding_index])

            if second_sharding_index is None:
                second_dim_spec = _DimSpec([])
            else:
                second_dim_spec = _DimSpec([second_sharding_index])

            sharding_sequence = [first_dim_spec, second_dim_spec]
            sharding_spec = ShardingSpec(device_mesh=device_mesh,
                                         entire_shape=entire_shape,
                                         sharding_sequence=sharding_sequence)
            strategies_vector_for_input.append(sharding_spec)
    setattr(nodes[1], 'strategies_vector', strategies_vector_for_input)

    # generate dot strategy
    strategies_vector = StrategiesVector(node=nodes[2])
    dot_handler = DotHandler(node=nodes[2],
                             device_mesh=device_mesh,
                             strategies_vector=strategies_vector,
                             shape_consistency_manager=shape_consistency_manager)
    strategies_vector = dot_handler.register_strategy()

    # ['S0S1 = S0R x RS1', 'S1S0 = S1R x RS0', 'S0R = S0S1 x S1R', 'S1R = S1S0 x S0R', 'RS1 = RS0 x S0S1', 'RS0 = RS1 x S1S0', 'RS0 = RR x RS0', 'RS1 = RR x RS1', 'RR = RR x RR']
    strategy_name_list = [strategy.name for strategy in strategies_vector]

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
