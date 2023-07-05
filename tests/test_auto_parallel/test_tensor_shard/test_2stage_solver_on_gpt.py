from time import time

import psutil
import pytest
import torch
import transformers

try:
    from colossalai._analyzer.fx.codegen import ActivationCheckpointCodeGen
    NON_CODEGEN = False
except:
    NON_CODEGEN = True

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.checkpoint.ckpt_solver_rotor import CheckpointSolverRotor
from colossalai.auto_parallel.passes.comm_metainfo_pass import comm_metainfo_pass
from colossalai.auto_parallel.passes.meta_info_prop import MetaInfoProp
from colossalai.auto_parallel.tensor_shard.initialize import (
    ModuleWrapper,
    build_strategy_constructor,
    solve_solution,
    transform_to_sharded_model,
)
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch, launch_from_torch
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.testing import parameterize, rerun_if_address_is_in_use, run_on_environment_flag, spawn
from tests.test_auto_parallel.test_tensor_shard.test_gpt.gpt_modules import GPT2MLP, GPT2Attention, GPT2Block, GPT2Model

BATCH_SIZE = 16
SEQ_LENGTH = 1024
HIDDEN_DIM = 2048
NUM_HEADS = 16
NUM_LAYERS = 2
VOCAB_SIZE = 50257
NUM_STEPS = 10


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def data_gen_mlp(model_cls):
    """
    Generate random data for resnet benchmarking
    """
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
    hidden_states = torch.rand((BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM), dtype=torch.float32)

    if model_cls == GPT2MLP:
        input_sample = (hidden_states.to('cuda'),)
        meta_input_sample = {
            'hidden_states': hidden_states.to('meta'),
        }
    elif model_cls in (GPT2Attention, GPT2Block):
        attention_mask = torch.zeros((1, SEQ_LENGTH), dtype=torch.int64)
        input_sample = (
            hidden_states.to('cuda'),
            attention_mask.to('cuda'),
        )
        meta_input_sample = {
            'hidden_states': hidden_states.to('meta'),
            'attention_mask': attention_mask.to('meta'),
        }
    else:
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        input_sample = (
            input_ids.to('cuda'),
            attention_mask.to('cuda'),
        )
        meta_input_sample = {
            'input_ids': input_ids.to('meta'),
            'attention_mask': attention_mask.to('meta'),
        }
    return input_sample, meta_input_sample


def check_2stage_solver_on_gpt(rank, world_size, port, model_cls):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    logger = get_dist_logger()
    config = transformers.GPT2Config(n_position=SEQ_LENGTH, n_layer=NUM_LAYERS, n_head=NUM_HEADS, n_embd=HIDDEN_DIM)
    if model_cls == GPT2MLP:
        model = model_cls(intermediate_size=4 * config.hidden_size, config=config).to('cuda')
    else:
        model = model_cls(config=config).to('cuda')

    input_sample, meta_input_sample = data_gen_mlp(model_cls)

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    tracer = ColoTracer(bias_addition_split=True)

    graph = tracer.trace(root=model, meta_args=meta_input_sample)
    graph.set_codegen(ActivationCheckpointCodeGen())
    gm = ColoGraphModule(model, graph, model.__class__.__name__)
    shape_prop_pass(gm, *meta_input_sample.values())
    gm.recompile()

    strategies_constructor = build_strategy_constructor(graph, device_mesh, 'standard', 'replicated', 'standard')
    solution = solve_solution(gm, strategies_constructor, memory_budget=-1)
    gm, sharding_spec_dicts = transform_to_sharded_model(gm, meta_input_sample, solution, device_mesh,
                                                         strategies_constructor)
    comm_metainfo_pass(gm, *sharding_spec_dicts)

    MetaInfoProp(gm).run()

    gm = ModuleWrapper(gm, *sharding_spec_dicts)
    ckpt_solver = CheckpointSolverRotor(gm.module.graph, 8 * 1024**3)
    gm.module.graph = ckpt_solver.solve()
    ckpt_solver.print_sequence()
    gm.module.recompile()
    print(gm.module)
    logger.info("*******************strategy selected*******************", ranks=[0])
    strategies_list = solution

    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
    for index, node in enumerate(nodes):
        logger.info(node.name, ranks=[0])
        logger.info(node.strategies_vector[strategies_list[index]].name, ranks=[0])

    optimizer = torch.optim.Adam(gm.parameters(), lr=0.01)
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    torch.cuda.synchronize()
    gm.train()

    for n in range(10):
        # we just use randomly generated data here
        input_sample, _ = data_gen_mlp(model_cls)
        mem_stamp0 = torch.cuda.memory_allocated(device='cuda:0') / 1024**2
        optimizer.zero_grad()
        start = time()
        loss = gm(*input_sample)
        loss.sum().backward()
        optimizer.step()
        # prof.step()
        torch.cuda.synchronize()
        step_time = time() - start
        logger.info(f"===============Round {n}===============", ranks=[0])
        logger.info(
            f"Peak Memory: {torch.cuda.max_memory_allocated(device='cuda:0') / 1024**2 - mem_stamp0} MB, Step Time: {step_time:.3f}s",
            ranks=[0])
    torch.cuda.synchronize()


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.skipif(NON_CODEGEN, reason='codegen is not available')
@pytest.mark.dist
@rerun_if_address_is_in_use()
@parameterize('model_cls', [GPT2MLP, GPT2Attention, GPT2Block, GPT2Model])
def test_2stage_solver_on_gpt(model_cls):
    spawn(
        check_2stage_solver_on_gpt,
        4,
        model_cls=model_cls,
    )


if __name__ == '__main__':
    test_2stage_solver_on_gpt()
