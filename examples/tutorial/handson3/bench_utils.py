import time
from functools import partial
from typing import Callable, Tuple

import numpy as np
import torch
import torchvision.models as tm

from colossalai.auto_parallel.checkpoint import CheckpointSolverRotor
from colossalai.fx import metainfo_trace


def bench(gm: torch.fx.GraphModule, criterion: torch.nn.Module, data_gen: Callable, num_steps: int = 5):
    gm.train()
    gm.cuda()
    step_time = float('inf')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cached = torch.cuda.max_memory_allocated(device="cuda")
    try:
        for _ in range(num_steps):
            args, label = data_gen()
            output, loss = None, None

            torch.cuda.synchronize(device="cuda")
            start = time.time()
            output = gm(*args)
            loss = criterion(output, label)
            loss.backward()
            torch.cuda.synchronize(device="cuda")
            step_time = min(step_time, time.time() - start)

            for child in gm.children():
                for param in child.parameters():
                    param.grad = None
            del args, label, output, loss
    except:
        del args, label, output, loss
    gm.to("cpu")
    torch.cuda.empty_cache()
    return (torch.cuda.max_memory_allocated(device="cuda") - cached) / 1024**2, step_time * 1.0e3


def bench_rotor(gm: torch.fx.GraphModule,
                criterion: torch.nn.Module,
                data_gen: Callable,
                num_steps: int = 5,
                sample_points: int = 20,
                free_memory: int = torch.cuda.mem_get_info()[0]):
    peak_hist, step_hist = [], []
    for budget in np.linspace(free_memory // 5, free_memory, sample_points):
        gm = metainfo_trace(gm, *data_gen()[0])
        solver = CheckpointSolverRotor(gm.graph, free_memory=budget)
        try:
            gm.graph = solver.solve()
            peak_memory, step_time = bench(gm,
                                           criterion,
                                           partial(data_gen, batch_size=2048, shape=(3, 224, 224)),
                                           num_steps=num_steps)
        except:
            peak_memory, step_time = budget / 1024**2, float('inf')
        peak_hist.append(peak_memory)
        step_hist.append(step_time)
    return peak_hist, step_hist
