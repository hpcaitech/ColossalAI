from argparse import ArgumentParser
from functools import partial

import matplotlib.pyplot as plt
import torch
import torchvision.models as tm
from bench_utils import GPTLMLoss, bench_rotor, data_gen_gpt2, data_gen_resnet, gpt2_medium

import colossalai
from colossalai.fx import metainfo_trace, symbolic_trace
from colossalai.testing import spawn


def _benchmark(rank, world_size, port, args):
    """
    Auto activation checkpoint solver benchmark, we provide benchmark on two models: gpt2_medium and resnet50.
    The benchmark will sample in a range of memory budget for each model and output the benchmark summary and
    data visualization of peak memory vs. budget memory and relative step time vs. peak memory.
    """
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    if args.model == "resnet50":
        model = tm.resnet50()
        data_gen = partial(data_gen_resnet, batch_size=128, shape=(3, 224, 224))
        gm = symbolic_trace(model)
        gm = metainfo_trace(gm, torch.empty(128, 3, 224, 224, device="meta"))
        loss = torch.nn.CrossEntropyLoss()
    else:
        model = gpt2_medium()
        data_gen = partial(data_gen_gpt2, batch_size=8, seq_len=1024, vocab_size=50257)
        data, mask = data_gen(device="meta")[0]
        gm = symbolic_trace(model, meta_args={"input_ids": data, "attention_mask": mask})
        gm = metainfo_trace(gm, data, mask)
        loss = GPTLMLoss()

    free_memory = 11000 * 1024**2 if args.model == "resnet50" else 56000 * 1024**2
    start_factor = 4 if args.model == "resnet50" else 10

    # trace and benchmark
    budgets, peak_hist, step_hist = bench_rotor(
        gm, loss, data_gen, num_steps=5, sample_points=15, free_memory=free_memory, start_factor=start_factor
    )

    # print summary
    print("==============benchmark summary==============")
    for budget, peak, step in zip(budgets, peak_hist, step_hist):
        print(f"memory budget: {budget:.3f} MB, peak memory: {peak:.3f} MB, step time: {step:.3f} MS")

    # plot valid results
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    valid_idx = step_hist.index(next(step for step in step_hist if step != float("inf")))

    # plot peak memory vs. budget memory
    axs[0].plot(budgets[valid_idx:], peak_hist[valid_idx:])
    axs[0].plot([budgets[valid_idx], budgets[-1]], [budgets[valid_idx], budgets[-1]], linestyle="--")
    axs[0].set_xlabel("Budget Memory (MB)")
    axs[0].set_ylabel("Peak Memory (MB)")
    axs[0].set_title("Peak Memory vs. Budget Memory")

    # plot relative step time vs. budget memory
    axs[1].plot(peak_hist[valid_idx:], [step_time / step_hist[-1] for step_time in step_hist[valid_idx:]])
    axs[1].plot([peak_hist[valid_idx], peak_hist[-1]], [1.0, 1.0], linestyle="--")
    axs[1].set_xlabel("Peak Memory (MB)")
    axs[1].set_ylabel("Relative Step Time")
    axs[1].set_title("Step Time vs. Peak Memory")
    axs[1].set_ylim(0.8, 1.5)

    # save plot
    fig.savefig(f"{args.model}_benchmark.png")


def auto_activation_checkpoint_benchmark(args):
    world_size = 1
    spawn(_benchmark, world_size, args=args)


if __name__ == "__main__":
    parser = ArgumentParser("Auto Activation Checkpoint Solver Benchmark")
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "resnet50"])
    args = parser.parse_args()

    auto_activation_checkpoint_benchmark(args)
