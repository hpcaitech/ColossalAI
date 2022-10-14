import torch
from torch.fx import GraphModule
import torch.nn as nn
import pytest

from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.device.device_mesh import DeviceMesh
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class ConvModel(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, condition, x, y):
        output = torch.where(condition, x, y)

        return output


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_where_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

    tracer = ColoTracer()
    model = ConvModel(16, 32)
    input_sample = {
        'condition': torch.rand(16, 32).to('meta'),
        'x': torch.rand(16, 32).to('meta'),
        'y': torch.rand(16, 32).to('meta')
    }
    # graph():
    #     %condition : torch.Tensor [#users=1] = placeholder[target=condition]
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %y : torch.Tensor [#users=1] = placeholder[target=y]
    #     %where : [#users=1] = call_function[target=torch.where](args = (%condition, %x, %y), kwargs = {})
    #     return where
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)

    # [condition, x, y, where, output]
    nodes = [node for node in gm.graph.nodes]
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    strategies_constructor.build_strategies_and_cost()
    strategy_map = strategies_constructor.strategy_map
    # check a tensor add with a scalar case
    where_node = strategy_map[nodes[3]]
    # ['[S0, S1] = [S0, S1] x [S0, S1] x [S0, S1]', '[S1, S0] = [S1, S0] x [S1, S0] x [S1, S0]', '[S01, R] = [S01, R] x [S01, R] x [S01, R]',
    #  '[R, S01] = [R, S01] x [R, S01] x [R, S01]', '[S0, R] = [S0, R] x [S0, R] x [S0, R]', '[R, S0] = [R, S0] x [R, S0] x [R, S0]',
    #  '[S1, R] = [S1, R] x [S1, R] x [S1, R]', '[R, S1] = [R, S1] x [R, S1] x [R, S1]', '[R, R] = [R, R] x [R, R] x [R, R]']
    assert len(where_node) == 9


if __name__ == '__main__':
    test_where_handler()
