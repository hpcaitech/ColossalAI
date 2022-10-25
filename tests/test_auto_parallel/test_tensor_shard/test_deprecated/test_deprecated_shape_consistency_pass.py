from functools import partial
import pytest
import torch
import torch.multiprocessing as mp
from torch.fx import GraphModule
import torch.nn as nn
import pytest
from colossalai.initialize import launch
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.logging import disable_existing_loggers
from colossalai.auto_parallel.tensor_shard.deprecated.cost_graph import CostGraph
from colossalai.auto_parallel.tensor_shard.deprecated.graph_analysis import GraphAnalyser
from colossalai.auto_parallel.tensor_shard.deprecated.strategies_constructor import StrategiesConstructor

from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.passes.experimental.adding_shape_consistency_pass import shape_consistency_pass, solution_annotatation_pass
from colossalai.auto_parallel.tensor_shard.deprecated import Solver
from colossalai.auto_parallel.tensor_shard.deprecated.options import SolverOptions
from colossalai.testing.pytest_wrapper import run_on_environment_flag


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


def check_apply(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    input = torch.rand(4, 4, 4, 4).cuda()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    entire_shape = torch.Size((4, 4, 8, 8))

    tracer = ColoTracer()
    model = ConvModel(4, 4).cuda()
    origin_output = model(input)
    input_sample = {'x': torch.rand(4, 4, 4, 4).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
    #     return conv
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    solver_options = SolverOptions(fast=True)
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    graph_analyser = GraphAnalyser(gm)
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])
    sharding_spec_dict, origin_spec_dict = solution_annotatation_pass(gm, solution, device_mesh)
    shape_consistency_pass(gm)
    gm.recompile()
    nodes = [node for node in gm.graph.nodes]
    # TODO: wrap the gm to avoid the influence of the user training code
    output = gm(input, sharding_spec_dict, origin_spec_dict)
    assert output.equal(origin_output)


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_apply():
    world_size = 4
    run_func = partial(check_apply, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_apply()
