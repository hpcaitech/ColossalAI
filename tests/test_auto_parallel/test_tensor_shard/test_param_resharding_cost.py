import torch

from colossalai.auto_parallel.tensor_shard.sharding_strategy import OperationDataType
from colossalai.auto_parallel.tensor_shard.solver import (
    CostGraph,
    GraphAnalyser,
    Solver,
    SolverOptions,
    StrategiesConstructor,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx import ColoGraphModule, ColoTracer
from colossalai.testing.pytest_wrapper import run_on_environment_flag


def _param_resharding_cost_assertion(node):
    for strategy in node.strategies_vector:
        for prev_node, resharding_cost in strategy.resharding_costs.items():
            if strategy.get_op_data_by_name(str(prev_node)).type == OperationDataType.PARAM:
                for cost in resharding_cost:
                    assert cost.fwd == 0
                    assert cost.bwd == 0
                    assert cost.total == 0


class LinearModel(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = x * 2

        return x


class ConvModel(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = x * 2

        return x


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_linear_module():
    model = LinearModel(4, 8)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    tracer = ColoTracer()
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %linear_weight : [#users=1] = get_attr[target=linear.weight]
    #     %linear_bias : [#users=1] = get_attr[target=linear.bias]
    #     %linear : [#users=1] = call_function[target=torch._C._nn.linear](args = (%x, %linear_weight), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%linear, %linear_bias), kwargs = {})
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%add, 2), kwargs = {})
    #     return mul
    graph = tracer.trace(root=model, meta_args={'x': torch.rand(4, 4).to('meta')})
    # def forward(self, x : torch.Tensor):
    #     linear_weight = self.linear.weight
    #     linear_bias = self.linear.bias
    #     linear = torch._C._nn.linear(x, linear_weight);  x = linear_weight = None
    #     add = linear + linear_bias;  linear = linear_bias = None
    #     mul = add * 2;  add = None
    #     return mul
    gm = ColoGraphModule(model, graph)
    gm.recompile()
    node_list = list(graph.nodes)

    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()
    linear_node = node_list[3]
    _param_resharding_cost_assertion(linear_node)


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_conv_module():
    model = ConvModel(3, 6, 2)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)
    tracer = ColoTracer()
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv_weight : [#users=1] = get_attr[target=conv.weight]
    #     %conv_bias : [#users=1] = get_attr[target=conv.bias]
    #     %conv2d : [#users=1] = call_function[target=torch.conv2d](args = (%x, %conv_weight), kwargs = {})
    #     %view : [#users=1] = call_method[target=view](args = (%conv_bias, [1, -1, 1, 1]), kwargs = {})
    #     %add : [#users=1] = call_function[target=operator.add](args = (%conv2d, %view), kwargs = {})
    #     %mul : [#users=1] = call_function[target=operator.mul](args = (%add, 2), kwargs = {})
    #     return mul
    graph = tracer.trace(root=model, meta_args={'x': torch.rand(4, 3, 64, 64).to('meta')})
    # def forward(self, x : torch.Tensor):
    #     conv_weight = self.conv.weight
    #     conv_bias = self.conv.bias
    #     conv2d = torch.conv2d(x, conv_weight);  x = conv_weight = None
    #     view = conv_bias.view([1, -1, 1, 1]);  conv_bias = None
    #     add = conv2d + view;  conv2d = view = None
    #     mul = add * 2;  add = None
    #     return mul
    gm = ColoGraphModule(model, graph)

    gm.recompile()
    node_list = list(graph.nodes)
    conv_node = node_list[3]
    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()
    _param_resharding_cost_assertion(conv_node)


if __name__ == '__main__':
    test_linear_module()
    test_conv_module()
