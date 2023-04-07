from time import time

import psutil
import pytest
import torch
import torchvision.models as tm

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
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.pytest_wrapper import run_on_environment_flag

BATCH_SIZE = 256


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def data_gen_resnet(batch_size, shape):
    """
    Generate random data for resnet benchmarking
    """
    data = torch.empty(batch_size, *shape, device=torch.cuda.current_device())
    label = torch.empty(batch_size, dtype=torch.long, device=torch.cuda.current_device()).random_(1000)
    return data, label


def check_2stage_solver_on_resnet(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    logger = get_dist_logger()
    model = tm.resnet50().cuda()

    meta_input_sample = {
        'x': torch.randn(BATCH_SIZE * 4, 3, 224, 224, device='meta'),
    }

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)

    tracer = ColoTracer(bias_addition_split=True, trace_act_ckpt=True)

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

    logger.info("*******************strategy selected*******************", ranks=[0])
    strategies_list = solution

    nodes = [strategies_vector.node for strategies_vector in strategies_constructor.leaf_strategies]
    for index, node in enumerate(nodes):
        logger.info(node.name, ranks=[0])
        logger.info(node.strategies_vector[strategies_list[index]].name, ranks=[0])

    # build criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(gm.parameters(), lr=0.01)
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    torch.cuda.synchronize()
    model.train()
    for n in range(10):
        # we just use randomly generated data here
        data, label = data_gen_resnet(BATCH_SIZE * 4, (3, 224, 224))
        mem_stamp0 = torch.cuda.memory_allocated(device='cuda:0') / 1024**2
        optimizer.zero_grad()
        start = time()
        outputs = gm(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
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
def test_2stage_solver_on_resnet():
    spawn(
        check_2stage_solver_on_resnet,
        4,
    )


if __name__ == '__main__':
    test_2stage_solver_on_resnet()
