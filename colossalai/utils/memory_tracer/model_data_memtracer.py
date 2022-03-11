from colossalai.utils.commons.singleton_meta import SingletonMeta
from colossalai.utils.memory_tracer.allocator import col_tensor_mem_usage
import torch


class ModelDataTracer(metaclass=SingletonMeta):
    """
    A singleton to trace model data usage during runtime.
    We have to trigger our API (trace_tensor, detach_tensor) when do model-data memory operation,
    including allocation, releasing and moving.
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
