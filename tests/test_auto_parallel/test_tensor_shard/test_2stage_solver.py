from pprint import pprint
from time import time

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torchvision.models as tm
from torch.fx import GraphModule

from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes import shape_prop_pass
# from colossalai.fx.tracer.tracer import ColoTracer
# from colossalai.fx import ColoGraphModule
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.checkpoint.ckpt_solver_rotor import CheckpointSolverRotor
from colossalai.auto_parallel.passes.comm_metainfo_pass import comm_metainfo_pass
from colossalai.auto_parallel.passes.meta_info_prop import MetaInfoProp
from colossalai.auto_parallel.passes.runtime_apply_pass import runtime_apply_pass
from colossalai.auto_parallel.passes.runtime_preparation_pass import runtime_preparation_pass
from colossalai.auto_parallel.tensor_shard.constants import BATCHNORM_MODULE_OP
from colossalai.auto_parallel.tensor_shard.options import DataloaderOption, ShardOption, SolverOptions, SolverPerference
from colossalai.auto_parallel.tensor_shard.sharding_strategy import ShardingSpec
from colossalai.auto_parallel.tensor_shard.solver import CostGraph, GraphAnalyser, Solver, StrategiesConstructor
from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch, launch_from_torch
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor.shape_consistency import ShapeConsistencyManager, to_global
from colossalai.testing import assert_close, assert_close_loose, parameterize, rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port
from tests.test_auto_parallel.test_tensor_shard.test_gpt.gpt_modules import GPT2LMHeadModel, GPTLMLoss

BATCH_SIZE = 128
SEQ_LENGTH = 128
HIDDEN_DIM = 4096
NUM_HEADS = 32
NUM_LAYERS = 4
VOCAB_SIZE = 50257
NUM_STEPS = 10


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


def main():
    disable_existing_loggers()
    launch_from_torch(config={})
    logger = get_dist_logger()
    model = tm.resnet50().cuda()

    meta_input_sample = {
        'x': torch.randn(128 * 4, 3, 224, 224, device='meta'),
    }

    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    shape_consistency_manager = ShapeConsistencyManager()

    tracer = ColoTracer(bias_addition_split=True, trace_act_ckpt=True)

    graph = tracer.trace(root=model, meta_args=meta_input_sample)
    gm = ColoGraphModule(model, graph, model.__class__.__name__)
    shape_prop_pass(gm, *meta_input_sample.values())
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
    gm: GraphModule
    gm = runtime_apply_pass(gm)
    gm.recompile()
    comm_metainfo_pass(gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
    MetaInfoProp(gm).run()
    ckpt_solver = CheckpointSolverRotor(gm.graph, 8 * 1024**3)
    gm.graph = ckpt_solver.solve()
    ckpt_solver.print_sequence()
    # assert False
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
        data, label = data_gen_resnet(128 * 4, (3, 224, 224))
        mem_stamp0 = torch.cuda.memory_allocated(device='cuda:0') / 1024**2
        optimizer.zero_grad()
        start = time()
        outputs = gm(data, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        # prof.step()
        torch.cuda.synchronize()
        step_time = time() - start
        logger.info(f"===============Round {n}===============", ranks=[0])
        logger.info(
            f"Peak Memory: {torch.cuda.max_memory_allocated(device='cuda:0') / 1024**2 - mem_stamp0} MB, Step Time: {step_time:.3f}s",
            ranks=[0])
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
