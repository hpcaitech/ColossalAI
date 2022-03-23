from colossalai.context.singleton_meta import SingletonMeta
from colossalai.utils.memory_tracer.commons import col_tensor_mem_usage
import torch


class ModelDataTracer(metaclass=SingletonMeta):
    """
    A tracer singleton to trace model data usage during runtime.
    The tracer is designed to trace the memory layout change during model-data tensors allocation, releasing, and moving.
    To achieve this goal, the developers have to call `ModelDataTracer` in the corresponding code explicitly.
    NOTE() now the class only trace cuda memory usage
    """

    def __init__(self) -> None:
        self._cuda_usage = 0

    def add_tensor(self, t: torch.Tensor):
        assert isinstance(t, torch.Tensor), f"ModelDataTracer add_tensor() should accept a torch.Tensor"
        mem_use = col_tensor_mem_usage(t)
        self._cuda_usage += mem_use

    def delete_tensor(self, t: torch.Tensor):
        assert isinstance(t, torch.Tensor), f"ModelDataTracer delete_tensor() should accept a torch.Tensor"
        mem_use = col_tensor_mem_usage(t)
        self._cuda_usage -= mem_use

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @property
    def cuda_usage(self):
        return self._cuda_usage


GLOBAL_MODEL_DATA_TRACER = ModelDataTracer()
