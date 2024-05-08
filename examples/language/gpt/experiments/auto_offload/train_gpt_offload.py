import argparse
import time

import pytest
import torch
from model_zoo import GPTLMLoss, get_gpt2_components
from torch.utils._pytree import tree_map

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.auto_parallel.offload.amp_optimizer import AMPOptimizer
from colossalai.auto_parallel.offload.mem_optimize import memory_optimize
from colossalai.auto_parallel.offload.solver import NOT_NVML
from colossalai.fx.profiler import parameter_size
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import spawn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gpt2_medium")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--solver_type", type=str, default="asyn")
    parser.add_argument("--memory_budget", type=float, default=16)
    return parser.parse_args()


@pytest.mark.skipif(NOT_NVML, reason="pynvml is not installed")
def train_gpt(args):
    memory_budget = args.memory_budget * 1024 * 1024 * 1024
    solver_type = args.solver_type
    model_type = args.model_type
    batch_size = args.batch_size

    # build model
    model_builder, data_gen = get_gpt2_components(model_type=model_type, batch_size=batch_size)
    label = torch.randint(
        low=0,
        high=128,
        size=(
            64,
            8,
        ),
        device=get_accelerator().get_current_device(),
    )
    criterion = GPTLMLoss()

    start_time = time.time()
    model = model_builder()
    model.train()
    param_size = parameter_size(model) / 1024**2 / 2
    init_time = time.time() - start_time
    print(f"init_param_size={param_size:.3f} MB | init_model_time={init_time:.3f} s")

    data_args = data_gen(device="cpu")
    wrap_fn = lambda x: x.to(dtype=torch.half) if isinstance(x, torch.Tensor) and torch.is_floating_point(x) else x
    data_args = tree_map(wrap_fn, data_args)
    start_time = time.time()
    model = memory_optimize(model, data_args, memory_budget, solver_type)
    solver_time = time.time() - start_time
    print(f"solver_time={solver_time:.3f} s")

    hybrid_optimizer = HybridAdam(model.model.parameters(), lr=1e-3)
    optim = AMPOptimizer(hybrid_optimizer, model)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    time_list = []
    data_args = data_gen(device="cuda")
    data_args = tree_map(wrap_fn, data_args)
    for step in range(10):
        optim.zero_grad()
        torch.cuda.synchronize()
        start_time = time.time()
        loss = criterion(model(**data_args), label)
        optim.backward(loss)
        torch.cuda.synchronize()
        time_list.append(time.time() - start_time)
        optim.step()

    torch.cuda.synchronize()

    exec_time = sum(sorted(time_list)[:5]) / 5
    runtime_peak_mem_alc = torch.cuda.max_memory_allocated() / 1024**2
    runtime_peak_mem_res = torch.cuda.max_memory_reserved() / 1024**2
    print(f"solver_type: {solver_type} | model_type: {model_type}")
    print(
        f"| exec_time={exec_time:.3f} s | param_size={param_size:.3f} MB "
        f"| runtime_peak_mem_alc={runtime_peak_mem_alc:.3f} MB| runtime_peak_mem_res={runtime_peak_mem_res:.3f} MB|"
    )
    print(time_list)


def run(rank, world_size, port, args):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    train_gpt(args)


if __name__ == "__main__":
    args = parse_args()
    spawn(run, 1, args=args)
