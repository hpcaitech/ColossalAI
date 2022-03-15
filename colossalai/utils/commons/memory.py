import imp
import torch
from colossalai.utils import get_current_device


def col_cuda_memory_capacity():
    """
    Get cuda memory capacity of the current cuda.
    """
    return torch.cuda.get_device_properties(get_current_device()).total_memory
