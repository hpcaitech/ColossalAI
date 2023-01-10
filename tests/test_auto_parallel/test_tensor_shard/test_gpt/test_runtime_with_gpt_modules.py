import copy
import random
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
from torch.fx import GraphModule

from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
from colossalai.auto_parallel.tensor_shard.sharding_strategy import ShardingSpec
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
from tests.test_auto_parallel.test_tensor_shard.test_gpt.gpt_modules import GPT2MLP, GPT2Attention, GPT2Block, GPT2Model

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


def _check_module_grad(module: torch.nn.Module, origin_param_dict: Dict[str, torch.Tensor],
                       best_sharding_spec_dict: Dict[str, ShardingSpec]):
    for name, param in module.named_parameters():
        param_grad = param.grad
        origin_param_grad = origin_param_dict[name].grad
        atoms = name.split('.')
        new_name = '_'.join(atoms)
        if new_name in best_sharding_spec_dict:
            param_sharding_spec = best_sharding_spec_dict[new_name]
            grad_to_compare = copy.deepcopy(param_grad)
            param_grad_global = to_global(grad_to_compare, param_sharding_spec)

            try:
                assert_close_loose(param_grad_global, origin_param_grad, rtol=1e-03, atol=1e-03)
            except:
                difference = param_grad_global - origin_param_grad
                avg_diff = difference.abs().sum() / difference.numel()
                assert avg_diff < 0.001
                print(f'{name} param has {avg_diff} average difference')


def check_attention_layer(rank, model_cls, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    config = transformers.GPT2Config(n_position=64, n_layer=1, n_head=16, n_embd=HIDDEN_DIM)

    if model_cls == GPT2MLP:
        model = model_cls(intermediate_size=4 * config.hidden_size, config=config).to('cuda')
    else:
        model = model_cls(config=config).to('cuda')
    test_model = copy.deepcopy(model)

    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    hidden_states = torch.rand((BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM), dtype=torch.float32)

    if model_cls == GPT2MLP:
        input_sample = (hidden_states.to('cuda'),)
        test_input_sample = copy.deepcopy(input_sample)
        meta_input_sample = {
            'hidden_states': hidden_states.to('meta'),
        }
    elif model_cls in (GPT2Attention, GPT2Block):
        input_sample = (
            hidden_states.to('cuda'),
            attention_mask.to('cuda'),
        )
        test_input_sample = copy.deepcopy(input_sample)
        meta_input_sample = {
            'hidden_states': hidden_states.to('meta'),
            'attention_mask': attention_mask.to('meta'),
        }
    else:
        input_sample = (
            input_ids.to('cuda'),
            attention_mask.to('cuda'),
        )
        test_input_sample = copy.deepcopy(input_sample)
        meta_input_sample = {
            'input_ids': input_ids.to('meta'),
            'attention_mask': attention_mask.to('meta'),
        }

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer()

    graph = tracer.trace(root=model, meta_args=meta_input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

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
    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
    best_sharding_spec_dict = {}
    for index, node in enumerate(nodes):
        best_sharding_spec_dict[node.name] = node.sharding_spec

    cuda_rng_state = torch.cuda.get_rng_state()
    cpu_rng_state = torch.get_rng_state()
    origin_output = test_model(*test_input_sample)
    torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(cpu_rng_state)
    output = gm(*input_sample, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
    assert_close(output, origin_output, rtol=1e-03, atol=1e-03)

    #*******************backward starting*******************
    cuda_rng_state = torch.cuda.get_rng_state()
    cpu_rng_state = torch.get_rng_state()
    output.sum().backward()
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)
    origin_output.sum().backward()
    origin_param_dict = dict(test_model.named_parameters())

    if rank == 0:
        print("*******************backward starting*******************")

    _check_module_grad(gm, origin_param_dict, best_sharding_spec_dict)

    if rank == 0:
        print("*******************backward finished*******************")

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
@parameterize('model_cls', [GPT2MLP, GPT2Block, GPT2Attention, GPT2Model])
@rerun_if_address_is_in_use()
def test_mlp_layer(model_cls):
    world_size = 4
    run_func = partial(check_attention_layer, model_cls=model_cls, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_mlp_layer()
