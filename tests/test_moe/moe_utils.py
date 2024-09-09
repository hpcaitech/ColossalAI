import os
import traceback
from contextlib import contextmanager
from time import sleep
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map


def assert_loose_close(a, b, dtype: torch.dtype = torch.float32, name=""):
    assert loose_close(a, b, dtype), f"{name} not close {a.mean()} {b.mean()}"


def loose_close(a, b, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3
    else:
        assert dtype is torch.float32
        rtol = 1e-05
        atol = 1e-08

    a = a.detach().to(dtype)
    b = b.detach().to(dtype).to(a.device)

    return torch.allclose(a, b, rtol=rtol, atol=atol)


def check_model_equal(model1, model2, dtype):
    assert set(model1.state_dict().keys()) == set(model2.state_dict().keys())
    for i, ((name, p1), p2) in enumerate(zip(model1.named_parameters(), model2.parameters())):
        assert_loose_close(p1, p2, dtype, name=name)


@contextmanager
def distributed_debug_mode(num_stacks: int = 1, funcs_to_patch: Optional[List[Callable]] = None, enable=True):
    if enable:
        assert (
            os.environ.get("CUDA_LAUNCH_BLOCKING", "0") == "1"
        ), f"Expect CUDA_LAUNCH_BLOCKING=1, got {os.environ.get('CUDA_LAUNCH_BLOCKING', '0')}"
    if funcs_to_patch is None:
        funcs_to_patch = [
            dist.all_reduce,
            dist.all_reduce_coalesced,
            dist.all_gather,
            dist.all_gather_coalesced,
            dist.all_gather_into_tensor,
            dist.all_to_all,
            dist.all_to_all_single,
            dist.reduce_scatter,
        ]

    original_funcs = {}
    patched_funcs = {}

    def make_patched(func):
        def patched_func(*args, **kwargs):
            stack = traceback.format_stack()

            def format_node(node):
                if isinstance(node, torch.Tensor):
                    return f"{node.shape}"
                elif isinstance(node, list):
                    return f"[{', '.join([format_node(n) for n in node])}]"

                return str(node)

            args_str, kwargs_str = tree_map(format_node, (args, kwargs))
            en = len(stack) - 1
            st = max(0, en - num_stacks)
            dist.barrier()
            sleep(0.001 * dist.get_rank())
            print(
                f"[Rank {dist.get_rank()}-{func.__name__}-{dist.get_process_group_ranks(kwargs.get('group', dist.group.WORLD))}]: Called from {''.join(stack[st:en])}args={args_str} kwargs={kwargs_str}\n"
            )
            dist.barrier()
            return func(*args, **kwargs)

        return patched_func

    if enable:
        for func in funcs_to_patch:
            original_funcs[func.__name__] = getattr(dist, func.__name__)
            patched_funcs[func.__name__] = make_patched(func)
            setattr(dist, func.__name__, patched_funcs[func.__name__])

    try:
        yield
    finally:
        for func_name, original_func in original_funcs.items():
            setattr(dist, func_name, original_func)
