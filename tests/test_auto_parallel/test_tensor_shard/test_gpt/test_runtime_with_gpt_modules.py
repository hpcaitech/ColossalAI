import copy
import random
from typing import Dict

import numpy as np
import pytest
import torch
import transformers
from torch.fx import GraphModule

from colossalai._analyzer.fx.passes.shape_prop import shape_prop_pass

# from colossalai.fx.tracer.tracer import ColoTracer
from colossalai._analyzer.fx.tracer.tracer import ColoTracer

try:
    from colossalai.auto_parallel.tensor_shard.initialize import (
        ModuleWrapper,
        build_strategy_constructor,
        solve_solution,
        transform_to_sharded_model,
    )

    NO_CODEGEN = False
except:
    NO_CODEGEN = True

from colossalai.auto_parallel.tensor_shard.sharding_strategy import ShardingSpec
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.shape_consistency import to_global
from colossalai.testing import assert_close, assert_close_loose, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.pytest_wrapper import run_on_environment_flag
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


def _check_module_grad(
    module: torch.nn.Module,
    origin_param_dict: Dict[str, torch.Tensor],
    best_sharding_spec_dict: Dict[str, ShardingSpec],
):
    for name, param in module.named_parameters():
        param_grad = param.grad
        name = name.replace("module.", "")
        origin_param_grad = origin_param_dict[name].grad
        atoms = name.split(".")
        new_name = "_".join(atoms)
        if new_name in best_sharding_spec_dict:
            param_sharding_spec = best_sharding_spec_dict[new_name]
            grad_to_compare = copy.deepcopy(param_grad)
            param_grad_global = to_global(grad_to_compare, param_sharding_spec)
            try:
                assert_close_loose(param_grad_global, origin_param_grad, rtol=1e-03, atol=1e-05)
            except:
                difference = param_grad_global - origin_param_grad
                avg_diff = difference.abs().sum() / difference.numel()
                assert avg_diff < 0.001
                print(f"{name} param has {avg_diff} average difference")


def check_attention_layer(rank, model_cls, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    config = transformers.GPT2Config(n_position=64, n_layer=2, n_head=16, n_embd=HIDDEN_DIM)

    if model_cls == GPT2MLP:
        model = model_cls(intermediate_size=4 * config.hidden_size, config=config).to("cuda")
    else:
        model = model_cls(config=config).to("cuda")
    test_model = copy.deepcopy(model)

    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    hidden_states = torch.rand((BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM), dtype=torch.float32)

    if model_cls == GPT2MLP:
        input_sample = (hidden_states.to("cuda"),)
        test_input_sample = copy.deepcopy(input_sample)
        meta_input_sample = {
            "hidden_states": hidden_states.to("meta"),
        }
    elif model_cls in (GPT2Attention, GPT2Block):
        input_sample = (
            hidden_states.to("cuda"),
            attention_mask.to("cuda"),
        )
        test_input_sample = copy.deepcopy(input_sample)
        meta_input_sample = {
            "hidden_states": hidden_states.to("meta"),
            "attention_mask": attention_mask.to("meta"),
        }
    else:
        input_sample = (
            input_ids.to("cuda"),
            attention_mask.to("cuda"),
        )
        test_input_sample = copy.deepcopy(input_sample)
        meta_input_sample = {
            "input_ids": input_ids.to("meta"),
            "attention_mask": attention_mask.to("meta"),
        }

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    tracer = ColoTracer(bias_addition_split=True)

    graph = tracer.trace(root=model, meta_args=meta_input_sample)
    gm = GraphModule(model, graph, model.__class__.__name__)
    shape_prop_pass(gm, *meta_input_sample.values())
    gm.recompile()

    strategies_constructor = build_strategy_constructor(graph, device_mesh, "standard", "replicated", "standard")
    solution = solve_solution(gm, strategies_constructor, memory_budget=-1)
    gm, sharding_spec_dicts = transform_to_sharded_model(
        gm, meta_input_sample, solution, device_mesh, strategies_constructor
    )
    gm = ModuleWrapper(gm, *sharding_spec_dicts)

    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
    best_sharding_spec_dict = {}
    for index, node in enumerate(nodes):
        best_sharding_spec_dict[node.name] = node.sharding_spec

    cuda_rng_state = torch.cuda.get_rng_state()
    cpu_rng_state = torch.get_rng_state()
    origin_output = test_model(*test_input_sample)
    torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(cpu_rng_state)
    output = gm(*input_sample)
    assert_close(output, origin_output, rtol=1e-03, atol=1e-03)

    # *******************backward starting*******************
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

    # *******************backward finished*******************

    # *******************strategy selected*******************
    if rank == 0:
        print("*******************strategy selected*******************")
        nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
        computation_cost = 0
        communication_cost = 0
        memory_cost = 0
        for index, node in enumerate(nodes):
            print(node.name, node.strategies_vector[solution[index]].name)
            computation_cost += node.strategies_vector[solution[index]].compute_cost.total
            communication_cost += node.strategies_vector[solution[index]].communication_cost.total
            node_memory_cost = node.strategies_vector[solution[index]].memory_cost.total
            if isinstance(node_memory_cost, tuple):
                node_memory_cost = node_memory_cost[0]
            memory_cost += node_memory_cost.activation + node_memory_cost.parameter

        print(f"computation cost is {computation_cost}")
        print(f"communication cost is {communication_cost}")
        print(f"memory cost is {memory_cost}")


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.skipif(NO_CODEGEN, reason="no codegen module")
@pytest.mark.dist
@parameterize("model_cls", [GPT2MLP, GPT2Block, GPT2Attention, GPT2Model])
@rerun_if_address_is_in_use()
def test_mlp_layer(model_cls):
    spawn(check_attention_layer, 4, model_cls=model_cls)


if __name__ == "__main__":
    test_mlp_layer()
