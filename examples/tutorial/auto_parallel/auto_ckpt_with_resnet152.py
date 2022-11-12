import time
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.models as tm
from bench_utils import bench

import colossalai
from colossalai.auto_parallel.checkpoint import CheckpointSolverRotor
from colossalai.fx import metainfo_trace, symbolic_trace
from colossalai.utils import free_port


def data_gen(batch_size, shape, device='cuda'):
    """
    Generate random data for benchmarking
    """
    data = torch.empty(batch_size, *shape, device=device)
    label = torch.empty(batch_size, dtype=torch.long, device=device).random_(1000)
    return (data,), label


def _resnet152_benchmark(rank, world_size, port, num_steps):
    """Resnet152 benchmark
    This benchmark test the through put of Resnet152 with our activation solver given the memory budget of 95% of
    maximum GPU memory, and with the batch size of [512, 1024, 2048]
    """
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = tm.resnet152()
    gm = symbolic_trace(model)
    raw_graph = deepcopy(gm.graph)
    peak_mems, through_puts, batch_sizes = [], [], [512, 1024, 2048]
    for batch_size in batch_sizes:
        batch_size = int(batch_size)
        gm = metainfo_trace(gm, torch.empty(batch_size, 3, 224, 224, device='meta'))
        solver = CheckpointSolverRotor(gm.graph, free_memory=torch.cuda.mem_get_info()[0] * 0.95)
        gm.graph = solver.solve()
        peak_mem, step_time = bench(gm,
                                    torch.nn.CrossEntropyLoss(),
                                    partial(data_gen, batch_size=batch_size, shape=(3, 224, 224)),
                                    num_steps=num_steps)
        peak_mems.append(peak_mem)
        through_puts.append(batch_size / step_time * 1.0e3)
        gm.graph = deepcopy(raw_graph)

    # print results
    print("===============test summary================")
    for batch_size, peak_mem, through_put in zip(batch_sizes, peak_mems, through_puts):
        print(f'batch_size: {int(batch_size)}, peak memory: {peak_mem:.3f} MB, through put: {through_put:.3f} images/s')

    plt.plot(batch_sizes, through_puts)
    plt.xlabel("batch size")
    plt.ylabel("through put (images/s)")
    plt.title("Resnet152 benchmark")
    plt.savefig("resnet152_benchmark.png")


def resnet152_benchmark(num_steps):
    world_size = 1
    run_func_module = partial(_resnet152_benchmark, world_size=world_size, port=free_port(), num_steps=num_steps)
    mp.spawn(run_func_module, nprocs=world_size)


if __name__ == "__main__":
    parser = ArgumentParser("ResNet152 Auto Activation Through Put Benchmark")
    parser.add_argument("--num_steps", type=int, default=5, help="number of test steps for benchmark, default 5")
    args = parser.parse_args()

    resnet152_benchmark(args.num_steps)
