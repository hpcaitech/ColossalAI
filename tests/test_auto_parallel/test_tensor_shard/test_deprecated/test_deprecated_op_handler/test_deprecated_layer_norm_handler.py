import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest
from colossalai.auto_parallel.tensor_shard.deprecated import sharding_strategy

from colossalai.fx.proxy import ColoProxy
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.tensor.sharding_spec import ShardingSpec, _DimSpec
from colossalai.auto_parallel.tensor_shard.deprecated.op_handler.layer_norm_handler import LayerNormHandler
from colossalai.auto_parallel.tensor_shard.deprecated.sharding_strategy import ShardingStrategy, StrategiesVector
from colossalai.device.device_mesh import DeviceMesh


class LNModel(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        x = x * 2
        x = self.ln(x)
        return x


def test_bn_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    entire_shape = torch.Size((4, 4, 128))

    tracer = ColoTracer()
    model = LNModel(128)
    input_sample = {'x': torch.rand(4, 4, 128).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%x, 2), kwargs = {})
    #     %ln : [#users=1] = call_module[target=ln](args = (%mul,), kwargs = {})
    #     return ln
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    # [x, mul, ln, output]
    nodes = [node for node in gm.graph.nodes]
    sharding_spec_for_input = ShardingSpec(device_mesh, entire_shape, {})
    sharding_strategy_for_input = ShardingStrategy('node_1', sharding_spec_for_input)
    strategies_vector_for_input = StrategiesVector(nodes[1])
    strategies_vector_for_input.append(sharding_strategy_for_input)
    setattr(nodes[1], 'strategies_vector', strategies_vector_for_input)

    # generate bn strategy
    strategies_vector = StrategiesVector(node=nodes[2])
    ln_handler = LayerNormHandler(
        node=nodes[2],
        device_mesh=device_mesh,
        strategies_vector=strategies_vector,
    )
    ln_handler.register_strategy()
    # ['[S0, R, R] = [S0, R, R] x [R]', '[R, S0, R] = [R, S0, R] x [R]', '[S1, R, R] = [S1, R, R] x [R]', '[R, S1, R] = [R, S1, R] x [R]',
    # '[S0, S1, R] = [S0, S1, R] x [R]', '[S1, S0, R] = [S1, S0, R] x [R]', '[S01, R, R] = [S01, R, R] x [R]', '[R, S01, R] = [R, S01, R] x [R]', 'RR = RR x R']
    strategy_name_list = [strategy.name for strategy in ln_handler.strategies_vector]

    assert len(strategy_name_list) == 9


if __name__ == '__main__':
    test_bn_handler()
