"""Get ``symbolic_profile()`` result with real execution"""
from typing import Callable

import torch
from torch.autograd.profiler_util import _format_memory, _format_time
from zoo import tm_models, tmm_models

from siu.fx.node_util import compute_size_in_bytes


def run_forward(gm: torch.fx.GraphModule, data_gen: Callable, num_steps: int):
    torch.cuda.reset_peak_memory_stats()
    forward_mem = -torch.cuda.memory_allocated(device="cuda:0")
    param_mem = -torch.cuda.memory_allocated(device="cuda:0")
    gm.cuda()
    param_mem += torch.cuda.memory_allocated(device="cuda:0")
    gm.train()
    for n in range(num_steps):
        torch.cuda.reset_peak_memory_stats()
        data, _ = data_gen

        # If we need to dive deep into the memory usage by
        # inspecting `saved_tensor_hooks`

        # =====================================================
        fwd_mem = 0
        ctx = set()

        def pack(x):
            if isinstance(x, torch.Tensor):
                nonlocal fwd_mem, ctx
                if x.data_ptr() not in ctx:
                    fwd_mem += compute_size_in_bytes(x)
                    ctx.add(x.data_ptr())
            return x

        def unpack(x):
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            output = gm(data)
        print(f'Memory estimation by saved_tensor_hooks: {_format_memory(fwd_mem)}')
        # =====================================================

        output = gm(data)
        forward_mem += torch.cuda.memory_allocated(device="cuda:0") / num_steps
        del output
    return forward_mem, param_mem


# TODO(syj): profile me
