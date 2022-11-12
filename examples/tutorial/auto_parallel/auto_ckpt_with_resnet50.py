import time
from argparse import ArgumentParser
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torchvision.models as tm
from bench_utils import bench_rotor

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


def _resnet50_benchmark(rank, world_size, port, batch_size, num_steps, sample_points, free_memory, start_factor):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = tm.resnet50()

    # trace and benchmark
    gm = symbolic_trace(model)
    gm = metainfo_trace(gm, torch.empty(batch_size, 3, 224, 224, device='meta'))
    budgets, peak_hist, step_hist = bench_rotor(gm,
                                                torch.nn.CrossEntropyLoss(),
                                                partial(data_gen, batch_size=batch_size, shape=(3, 224, 224)),
                                                num_steps=num_steps,
                                                sample_points=sample_points,
                                                free_memory=free_memory,
                                                start_factor=start_factor)

    # print summary
    print("==============test summary==============")
    for budget, peak, step in zip(budgets, peak_hist, step_hist):
        print(f'memory budget: {budget:.3f} MB, peak memory: {peak:.3f} MB, step time: {step:.3f} MS')

    # plot valid results
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    valid_idx = step_hist.index(next(step for step in step_hist if step != float("inf")))

    # plot peak memory vs. budget memory
    axs[0].plot(budgets[valid_idx:], peak_hist[valid_idx:])
    axs[0].plot([budgets[valid_idx], budgets[-1]], [budgets[valid_idx], budgets[-1]], linestyle='--')
    axs[0].set_xlabel("Budget Memory (MB)")
    axs[0].set_ylabel("Peak Memory (MB)")
    axs[0].set_title("Peak Memory vs. Budget Memory")

    # plot relative step time vs. budget memory
    axs[1].plot(peak_hist[valid_idx:], [step_time / step_hist[-1] for step_time in step_hist[valid_idx:]])
    axs[1].plot([peak_hist[valid_idx], peak_hist[-1]], [1.0, 1.0], linestyle='--')
    axs[1].set_xlabel("Peak Memory (MB)")
    axs[1].set_ylabel("Relative Step Time")
    axs[1].set_title("Step Time vs. Peak Memory")
    axs[1].set_ylim(0.8, 1.5)

    # save plot
    fig.savefig("resnet50_benchmark.png")


def resnet50_benchmark(batch_size, num_steps, sample_points, free_memory, start_factor):
    world_size = 1
    run_func_module = partial(_resnet50_benchmark,
                              world_size=world_size,
                              port=free_port(),
                              batch_size=batch_size,
                              num_steps=num_steps,
                              sample_points=sample_points,
                              free_memory=free_memory,
                              start_factor=start_factor)
    mp.spawn(run_func_module, nprocs=world_size)


if __name__ == "__main__":
    parser = ArgumentParser("ResNet50 Auto Activation Benchmark")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for benchmark, default 128")
    parser.add_argument("--num_steps", type=int, default=5, help="number of test steps for benchmark, default 5")
    parser.add_argument(
        "--sample_points",
        type=int,
        default=15,
        help=
        "number of sample points for benchmark from start memory budget to maximum memory budget (free_memory), default 15"
    )
    parser.add_argument("--free_memory",
                        type=int,
                        default=11000,
                        help="maximum memory budget in MB for benchmark, default 11000 MB")
    parser.add_argument(
        "--start_factor",
        type=int,
        default=4,
        help=
        "start memory budget factor for benchmark, the start memory budget will be free_memory / start_factor, default 4"
    )
    args = parser.parse_args()

    resnet50_benchmark(args.batch_size, args.num_steps, args.sample_points, args.free_memory * 1024**2,
                       args.start_factor)
