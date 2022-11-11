import time
from argparse import ArgumentParser
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torchvision.models as tm
from bench_utils import GPTLMLoss, bench_rotor, gpt2_medium

import colossalai
from colossalai.auto_parallel.checkpoint import CheckpointSolverRotor
from colossalai.fx import metainfo_trace, symbolic_trace
from colossalai.utils import free_port


def data_gen(batch_size, seq_len, vocab_size, device='cuda:0'):
    """
    Generate random data for benchmarking
    """
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    return (input_ids, attention_mask), attention_mask


def _gpt2_benchmark(rank, world_size, port, batch_size, num_steps, sample_points, free_memory, start_factor):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = gpt2_medium()

    # trace and benchmark
    data, mask = data_gen(batch_size, 1024, 50257, device='meta')[0]
    gm = symbolic_trace(model, meta_args={'input_ids': data, 'attention_mask': mask})
    gm = metainfo_trace(gm, data, mask)
    budgets, peak_hist, step_hist = bench_rotor(gm,
                                                GPTLMLoss(),
                                                partial(data_gen, batch_size=batch_size, seq_len=1024,
                                                        vocab_size=50257),
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
    fig.savefig("gpt2_benchmark.png")


def gpt2_benchmark(batch_size, num_steps, sample_points, free_memory, start_factor):
    world_size = 1
    run_func_module = partial(_gpt2_benchmark,
                              world_size=world_size,
                              port=free_port(),
                              batch_size=batch_size,
                              num_steps=num_steps,
                              sample_points=sample_points,
                              free_memory=free_memory,
                              start_factor=start_factor)
    mp.spawn(run_func_module, nprocs=world_size)


if __name__ == "__main__":
    parser = ArgumentParser("GPT2 medium Auto Activation Benchmark")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for benchmark, default 8")
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
                        default=56000,
                        help="maximum memory budget in MB for benchmark, default 56000 MB")
    parser.add_argument(
        "--start_factor",
        type=int,
        default=10,
        help=
        "start memory budget factor for benchmark, the start memory budget will be free_memory / start_factor, default 10"
    )
    args = parser.parse_args()

    gpt2_benchmark(args.batch_size, args.num_steps, args.sample_points, args.free_memory * 1024**2, args.start_factor)
