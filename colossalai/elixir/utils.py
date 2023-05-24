import contextlib
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def seed_all(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:    # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def init_distributed():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])

    init_method = f'tcp://[{host}]:{port}'
    dist.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=world_size)

    # set cuda device
    if torch.cuda.is_available():
        # if local rank is not given, calculate automatically
        torch.cuda.set_device(local_rank)

    seed_all(1024)


def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
        dist.barrier()
    else:
        print(*args, **kwargs)


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def calc_buffer_size(m: nn.Module, test_dtype: torch.dtype = torch.float):
    max_sum_size = 0
    for module in m.modules():
        sum_p_size = 0
        for param in module.parameters(recurse=False):
            assert param.dtype == test_dtype
            sum_p_size += param.numel()
        max_sum_size = max(max_sum_size, sum_p_size)
    return max_sum_size


def calc_block_usage():
    snap_shot = torch.cuda.memory_snapshot()

    total_sum = 0
    active_sum = 0
    for info_dict in snap_shot:
        blocks = info_dict.get('blocks')
        for b in blocks:
            size = b.get('size')
            state = b.get('state')
            total_sum += size
            if state == 'active_allocated':
                active_sum += size

    active_ratio = 1
    if total_sum > 0:
        active_ratio = active_sum / total_sum

    print(f'memory snap shot: active ratio {active_ratio:.2f}')
