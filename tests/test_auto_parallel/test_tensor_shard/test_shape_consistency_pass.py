import copy
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.fx import GraphModule

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.solver import (
    CostGraph,
    GraphAnalyser,
    Solver,
    SolverOptions,
    StrategiesConstructor,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.tracer.tracer import ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import assert_close, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port


class ConvModel(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x)
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

    tracer = ColoTracer()
    model = ConvModel(4, 4).cuda()
    test_model = copy.deepcopy(model)
    test_input = copy.deepcopy(input)

    input_sample = {'x': torch.rand(4, 4, 4, 4).to('meta')}
    # graph():
    #     %x : torch.Tensor [#users=1] = placeholder[target=x]
    #     %conv : [#users=1] = call_module[target=conv](args = (%mul,), kwargs = {})
    #     return conv
    graph = tracer.trace(root=model, meta_args=input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
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
    nodes = [node for node in gm.graph.nodes]
    # TODO: wrap the gm to avoid the influence of the user training code
    output = gm(input, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
    origin_output = test_model(test_input)
    assert output.equal(origin_output)
    origin_loss = origin_output.sum()
    loss = output.sum()

    origin_loss.backward()
    loss.backward()

    grad_0 = test_model.conv.weight.grad.narrow(0, 0, 2)
    grad_1 = test_model.conv.weight.grad.narrow(0, 2, 2)

    if rank in (0, 1):
        assert_close(gm.conv.weight.grad.data, grad_0.data)
    elif rank in (2, 3):
        assert_close(gm.conv.weight.grad.data, grad_1.data)


# skip this test due to pulp not installed in CI environment
@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_apply():
    world_size = 4
    run_func = partial(check_apply, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_apply()
