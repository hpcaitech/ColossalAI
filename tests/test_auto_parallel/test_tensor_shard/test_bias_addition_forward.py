from functools import partial

import pytest
import torch
import torch.multiprocessing as mp

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
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
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, assert_close_loose, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port


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


def check_linear_module(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = LinearModel(4, 8).cuda()
    input = torch.rand(4, 4).cuda()
    output_compare = model(input)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
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
    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    graph_analyser = GraphAnalyser(gm)
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])
    gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict = runtime_preparation_pass(gm, solution, device_mesh)

    gm = runtime_apply_pass(gm)
    gm.recompile()
    output = gm(input, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
    assert_close(output, output_compare)


def check_conv_module(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = ConvModel(3, 6, 2).cuda()
    input = torch.rand(4, 3, 64, 64).cuda()
    output_compare = model(input)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
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

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    graph_analyser = GraphAnalyser(gm)
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])

    gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict = runtime_preparation_pass(gm, solution, device_mesh)

    gm = runtime_apply_pass(gm)
    gm.recompile()
    output = gm(input, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
    assert_close(output, output_compare)


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bias_addition_module():
    world_size = 4
    run_func_linear = partial(check_linear_module, world_size=world_size, port=free_port())
    mp.spawn(run_func_linear, nprocs=world_size)
    run_func_conv = partial(check_conv_module, world_size=world_size, port=free_port())
    mp.spawn(run_func_conv, nprocs=world_size)


if __name__ == '__main__':
    test_bias_addition_module()
