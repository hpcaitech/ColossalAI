import copy
import random
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from torch.fx import GraphModule
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers.pytorch_utils import Conv1D

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
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
from colossalai.tensor.shape_consistency import ShapeConsistencyManager, to_global
from colossalai.testing import assert_close, assert_close_loose, parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port

BATCH_SIZE = 1
SEQ_LENGTH = 32
HIDDEN_DIM = 768

seed = 128
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GPT2MLP(nn.Module):

    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        # We temporarily banned the Dropout layer because the rng state need
        # to process to get the correct result.
        # self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        # TODO: the rng state need to be fixed for distributed runtime
        # hidden_states = self.dropout(hidden_states)
        return hidden_states


def check_mlp_layer(rank, model_cls, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    config = transformers.GPT2Config(n_position=64, n_layer=4, n_head=16, n_embd=HIDDEN_DIM)
    model = model_cls(intermediate_size=4 * config.hidden_size, config=config).to('cuda')
    input = torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to('cuda')
    test_model = copy.deepcopy(model)
    test_input = copy.deepcopy(input)
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()

    input_sample = {
        'hidden_states': torch.rand(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM).to('meta'),
    }

    graph = tracer.trace(root=model, meta_args=input_sample)
    print(graph)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()
    print(gm)
    graph_analyser = GraphAnalyser(gm)
    liveness_list = graph_analyser.liveness_analysis()
    solver_options = SolverOptions()
    strategies_constructor = StrategiesConstructor(graph, device_mesh, solver_options)
    strategies_constructor.build_strategies_and_cost()

    cost_graph = CostGraph(strategies_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategies_constructor, cost_graph, graph_analyser, memory_budget=-1)
    ret = solver.call_solver_serialized_args()

    solution = list(ret[0])
    gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict = runtime_preparation_pass(
        gm, solution, device_mesh, strategies_constructor)
    gm = runtime_apply_pass(gm)
    gm.recompile()
    cuda_rng_state = torch.cuda.get_rng_state()
    cpu_rng_state = torch.get_rng_state()
    origin_output = test_model(test_input)
    torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(cpu_rng_state)
    output = gm(input, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
    assert_close(output, origin_output, rtol=1e-03, atol=1e-04)

    #*******************backward starting*******************
    cuda_rng_state = torch.cuda.get_rng_state()
    output.sum().backward()
    torch.cuda.set_rng_state(cuda_rng_state)
    origin_output.sum().backward()
    origin_param_dict = dict(test_model.named_parameters())
    if rank == 0:
        print("*******************backward starting*******************")
        for name, param in model.named_parameters():
            param_grad = param.grad
            origin_param_grad = origin_param_dict[name].grad
            origin_param_size = origin_param_grad.shape[-1]
            print(name, param_grad, origin_param_grad)
            if name == 'c_fc.bias':
                assert_close_loose(param_grad,
                                   origin_param_grad.narrow(0, 0, origin_param_size // 2),
                                   rtol=1e-03,
                                   atol=1e-03)
            else:
                assert_close_loose(param_grad, origin_param_grad, rtol=1e-03, atol=1e-03)
        print("*******************backward finished*******************")
    if rank == 1:
        for name, param in model.named_parameters():
            param_grad = param.grad
            origin_param_grad = origin_param_dict[name].grad
            origin_param_size = origin_param_grad.shape[-1]
            if name == 'c_fc.bias':
                assert_close_loose(param_grad,
                                   origin_param_grad.narrow(0, origin_param_size // 2, origin_param_size // 2),
                                   rtol=1e-03,
                                   atol=1e-03)
            else:
                assert_close_loose(param_grad, origin_param_grad, rtol=1e-03, atol=1e-03)
    if rank == 2:
        for name, param in model.named_parameters():
            param_grad = param.grad
            origin_param_grad = origin_param_dict[name].grad
            origin_param_size = origin_param_grad.shape[-1]
            if name == 'c_fc.bias':
                assert_close_loose(param_grad,
                                   origin_param_grad.narrow(0, 0, origin_param_size // 2),
                                   rtol=1e-03,
                                   atol=1e-03)
            else:
                assert_close_loose(param_grad, origin_param_grad, rtol=1e-03, atol=1e-03)
    if rank == 3:
        for name, param in model.named_parameters():
            param_grad = param.grad
            origin_param_grad = origin_param_dict[name].grad
            origin_param_size = origin_param_grad.shape[-1]
            if name == 'c_fc.bias':
                assert_close_loose(param_grad,
                                   origin_param_grad.narrow(0, origin_param_size // 2, origin_param_size // 2),
                                   rtol=1e-03,
                                   atol=1e-03)
            else:
                assert_close_loose(param_grad, origin_param_grad, rtol=1e-03, atol=1e-03)

    #*******************backward finished*******************

    #*******************strategy selected*******************
    if rank == 0:
        print("*******************strategy selected*******************")
        strategies_list = solver.last_s_val
        nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
        computation_cost = 0
        communication_cost = 0
        memory_cost = 0
        for index, node in enumerate(nodes):
            print(node.name, node.strategies_vector[strategies_list[index]].name)
            computation_cost += node.strategies_vector[strategies_list[index]].compute_cost.total
            communication_cost += node.strategies_vector[strategies_list[index]].communication_cost.total
            node_memory_cost = node.strategies_vector[strategies_list[index]].memory_cost.total
            if isinstance(node_memory_cost, tuple):
                node_memory_cost = node_memory_cost[0]
            memory_cost += node_memory_cost.activation + node_memory_cost.parameter

        print(f'computation cost is {computation_cost}')
        print(f'communication cost is {communication_cost}')
        print(f'memory cost is {memory_cost}')


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@parameterize('model_cls', [GPT2MLP])
@rerun_if_address_is_in_use()
def test_mlp_layer(model_cls):
    world_size = 4
    run_func = partial(check_mlp_layer, model_cls=model_cls, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_mlp_layer()
