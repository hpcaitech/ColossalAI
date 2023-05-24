from functools import cache

import torch
from torch.cuda._utils import _get_device_index

elixir_cuda_fraction = dict()


@cache
def gpu_device():
    return torch.device(torch.cuda.current_device())


def set_memory_fraction(fraction, device=None):
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    elixir_cuda_fraction[device] = fraction


def get_allowed_memory(device=None):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    fraction = elixir_cuda_fraction.get(device, 1.0)
    return int(fraction * total_memory)
