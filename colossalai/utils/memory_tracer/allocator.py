import torch
from colossalai.utils.commons.singleton_meta import SingletonMeta
from colossalai.zero.sharded_param import ShardedTensor

from typing import Union


def col_tensor_mem_usage(t: Union[torch.Tensor, ShardedTensor]) -> int:
    if isinstance(t, ShardedTensor):
        target = t.payload
    else:
        target = t
    return target.numel() * target.element_size()


class ModelDataTracer(metaclass=SingletonMeta):
    """
    A singleton to trace model data usage during runtime.
    """

    def __init__(self) -> None:
        self._cpu_usage = 0
        self._cuda_usage = 0

    def trace_tensor(self, t: torch.Tensor):
        mem_use = col_tensor_mem_usage(t)
        if t.device.type == 'cpu':
            self._cpu_usage += mem_use
        elif t.device.type == 'cuda':
            self._cuda_usage += mem_use
        else:
            raise RuntimeError

    def detach_tensor(self, t: torch.Tensor):
        mem_use = col_tensor_mem_usage(t)
        if t.device.type == 'cpu':
            self._cpu_usage -= mem_use
        elif t.device.type == 'cuda':
            self._cuda_usage -= mem_use
        else:
            raise RuntimeError

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @property
    def cuda_usage(self):
        return self._cuda_usage


GLOBAL_MODEL_DATA_TRACER = ModelDataTracer()


def col_allocate_payload(device: torch.device) -> torch.Tensor:
    pass


def col_release_payload(t: torch.Tensor):
    pass
