import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
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
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv_weight : [#users=1] = get_attr[target=conv.weight]
    #     %conv_bias : [#users=1] = get_attr[target=conv.bias]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%x, %conv_weight), kwargs = {groups: 1, dilation: (1, 1), stride: (1, 1), padding: (0, 0)})
    #     %view : [#users=1] = call_method[target=view](args = (%conv_bias, [1, -1, 1, 1]), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%conv2d, %view), kwargs = {})
    #     %flatten : [#users=1] = call_function[target=torch.flatten](args = (%add,), kwargs = {})
    #     return flatten
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    # [x, conv, flatten, output]
    nodes = [node for node in gm.graph.nodes]
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    strategies_constructor.build_strategies_and_cost()
    strategy_map = strategies_constructor.strategy_map
    add_strategies = strategy_map[nodes[5]]
    flatten_strategies = strategy_map[nodes[6]]
    flatten_strategies_cover_list = [strategy.input_shardings[0].sharding_sequence for strategy in flatten_strategies]
    for strategy in add_strategies:
        assert strategy.output_sharding_spec.sharding_sequence in flatten_strategies_cover_list


if __name__ == '__main__':
    test_conv_handler()
