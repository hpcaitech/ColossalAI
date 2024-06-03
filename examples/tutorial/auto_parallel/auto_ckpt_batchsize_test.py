from copy import deepcopy
from functools import partial

import torch
import torchvision.models as tm
from bench_utils import bench, data_gen_resnet

import colossalai
from colossalai.auto_parallel.checkpoint import CheckpointSolverRotor
from colossalai.fx import metainfo_trace, symbolic_trace
from colossalai.testing import spawn


def _benchmark(rank, world_size, port):
    """Auto activation checkpoint batchsize benchmark
    This benchmark test the through put of Resnet152 with our activation solver given the memory budget of 95% of
    maximum GPU memory, and with the batch size of [512, 1024, 2048], you could see that using auto activation
    checkpoint with optimality guarantee, we might be able to find better batch size for the model, as larger batch
    size means that we are able to use larger portion of GPU FLOPS, while recomputation scheduling with our solver
    only result in minor performance drop. So at last we might be able to find better training batch size for our
    model (combine with large batch training optimizer such as LAMB).
    """
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = tm.resnet152()
    gm = symbolic_trace(model)
    raw_graph = deepcopy(gm.graph)
    peak_mems, through_puts, batch_sizes = [], [], [512, 1024, 2048]
    for batch_size in batch_sizes:
        batch_size = int(batch_size)
        gm = metainfo_trace(gm, torch.empty(batch_size, 3, 224, 224, device="meta"))
        solver = CheckpointSolverRotor(gm.graph, free_memory=torch.cuda.mem_get_info()[0] * 0.95)
        gm.graph = solver.solve()
        peak_mem, step_time = bench(
            gm,
            torch.nn.CrossEntropyLoss(),
            partial(data_gen_resnet, batch_size=batch_size, shape=(3, 224, 224)),
            num_steps=5,
        )
        peak_mems.append(peak_mem)
        through_puts.append(batch_size / step_time * 1.0e3)
        gm.graph = deepcopy(raw_graph)

    # print results
    print("===============benchmark summary================")
    for batch_size, peak_mem, through_put in zip(batch_sizes, peak_mems, through_puts):
        print(f"batch_size: {int(batch_size)}, peak memory: {peak_mem:.3f} MB, through put: {through_put:.3f} images/s")


def auto_activation_checkpoint_batchsize_benchmark():
    spawn(_benchmark, 1)


if __name__ == "__main__":
    auto_activation_checkpoint_batchsize_benchmark()
