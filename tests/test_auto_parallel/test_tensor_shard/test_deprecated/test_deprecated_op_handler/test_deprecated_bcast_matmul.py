import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class MatmulModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = torch.matmul(x1, x2)

        return x


@run_on_environment_flag(name='AUTO_PARALLEL')
def test_conv_handler():
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape)

    tracer = ColoTracer()
    model = MatmulModel()
    input_sample = {'x1': torch.rand(4, 4, 8).to('meta'), 'x2': torch.rand(4, 1, 8, 4).to('meta')}
    # graph():
    #     %x1 : torch.Tensor [#users=1] = placeholder[target=x1]
    #     %x2 : torch.Tensor [#users=1] = placeholder[target=x2]
    #     %matmul : [#users=1] = call_function[target=torch.matmul](args = (%x1, %x2), kwargs = {})
    #     return matmul
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    # [x1, x2, matmul, output]
    nodes = [node for node in gm.graph.nodes]
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)

    strategies_constructor.build_strategies_and_cost()
    strategy_map = strategies_constructor.strategy_map
    matmul_strategies = strategy_map[nodes[2]]
    assert len(matmul_strategies) == 30


if __name__ == '__main__':
    test_conv_handler()
