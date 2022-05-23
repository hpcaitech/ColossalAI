import math
import time
import torch

from colossalai.utils import MultiTimer
from colossalai.context import ParallelMode, Config
from typing import List, Dict, Tuple, Callable


def get_time_stamp() -> int:
    """
    Return the time stamp for profiling.

    Returns:
        time_stamp (int): the time given by time.time()
    """

    torch.cuda.synchronize()
    time_stamp = time.time()
    return time_stamp


def get_memory_states() -> Tuple[float]:
    """
    Return the memory statistics.

    Returns:
        max_allocated (float): the allocated CUDA memory 
        max_cached (float):  the cached CUDA memory 
    """

    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    max_cached = torch.cuda.max_memory_reserved() / (1024**3)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return max_allocated, max_cached


def find_all_configs(device_cnt: int) -> List[Dict]:
    """
    Find all possible configurations for tensor parallelism

    Args:
        device_cnt (int): the number of devices

    Returns:
        config_list (List[Dict]): a list of configurations
    """

    def _is_square(num):
        # 2D parallel should be implemented with at least 2 devices.
        if num <= 1:
            return False
        return math.floor(math.sqrt(num))**2 == num

    def _is_cube(num):
        # 3D parallel should be implemented with at least 2 devices.
        if num <= 1:
            return False
        return math.floor(num**(1. / 3.))**3 == num

    config_list = []

    # add non-parallel config
    config = dict(parallel=dict(tensor=dict(size=device_cnt, mode=None)))
    config_list.append(config)

    # add 1D config
    config = dict(parallel=dict(tensor=dict(size=device_cnt, mode='1d')))
    config_list.append(config)

    # add 2D config only if device_cnt is a square
    if _is_square(device_cnt):
        config = dict(parallel=dict(tensor=dict(size=device_cnt, mode='2d')))
        config_list.append(config)

    # check for 2.5D
    # iterate over depth
    for depth in range(1, device_cnt):
        if device_cnt % depth == 0 and _is_square(device_cnt // depth):
            config = dict(parallel=dict(tensor=dict(size=device_cnt, mode='2.5d', depth=depth)))
            config_list.append(config)

    # check for 3D if device_cnt is a cube
    if _is_cube(device_cnt):
        config = dict(parallel=dict(tensor=dict(size=device_cnt, mode='3d')))
        config_list.append(config)

    config_list = [Config(cfg) for cfg in config_list]
    return config_list


def profile_model(model: torch.nn.Module, warmup_steps: int, profile_steps: int, data_func: Callable,
                  timer: MultiTimer) -> Tuple[float]:
    """
    Profile the forward and backward of a model

    Args:
        model (torch.nn.Module): a PyTorch model
        warmup_steps (int): the number of steps for warmup
        profile_steps (int): the number of steps for profiling
        data_func (Callable): a function to generate random data
        timer (colossalai.utils.Multitimer): a timer instance for time recording
    
    Returns:
        fwd_time (float): the average forward time taken by forward pass in second
        bwd_time (float): the average backward time taken by forward pass in second
        max_allocated (float): the maximum GPU memory allocated in GB
        max_cached (float): the maximum GPU memory cached in GB
    """

    def _run_step(data):
        timer.start('forward')
        out = model(data)
        timer.stop('forward', keep_in_history=True)
        timer.start('backward')
        out.mean().backward()
        timer.stop('backward', keep_in_history=True)

    data_list = [data_func() for _ in range(warmup_steps)]
    for data in data_list:
        _run_step(data)
    timer.reset('forward')
    timer.reset('backward')

    for _ in range(profile_steps):
        data = data_func()
        _run_step(data)

    max_allocated, max_cached = get_memory_states()
    fwd_time = timer.get_timer('forward').get_history_mean()
    bwd_time = timer.get_timer('backward').get_history_mean()
    return fwd_time, bwd_time, max_allocated, max_cached


def get_batch_data(dim: int, batch_size: int, seq_length: int, mode: ParallelMode) -> torch.Tensor:
    """
    Return a random data of shape (batch_size, seq_length, dim) for profiling.

    Args:
        dim (int): hidden size
        batch_size (int): the number of data samples
        seq_length (int): the number of tokens
        mode (ParallelMode): Colossal-AI ParallelMode enum

    Returns:
        data (torch.Tensor): random data
    """

    if mode in ['2d', '2.5d']:
        batch_size = batch_size // 2
        dim = dim // 2
    elif mode == '3d':
        batch_size = batch_size // 4
        dim = dim // 2

    data = torch.rand(batch_size, seq_length, dim).cuda()
    return data
