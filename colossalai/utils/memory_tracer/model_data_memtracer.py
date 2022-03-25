from colossalai.context.singleton_meta import SingletonMeta
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
import torch
from typing import Union


def _col_tensor_mem_usage(t: Union[torch.Tensor, ShardedTensor]) -> int:
    if isinstance(t, ShardedTensor):
        target = t.payload
    else:
        target = t
    return target.numel() * target.element_size()


class ModelDataTracer(metaclass=SingletonMeta):
    """
    A tracer singleton to trace model data usage during runtime.
    The tracer is designed to trace the memory layout change during model-data tensors allocation, releasing, and moving.
    To achieve this goal, the developers have to call `ModelDataTracer` in the corresponding code explicitly.
    NOTE() now the class only trace cuda memory usage
    """

    def __init__(self) -> None:
        self._cuda_usage = 0
        self._cpu_usage = 0
        self._start_flag = False

    def start(self) -> None:
        self._start_flag = True

    def close(self) -> None:
        self._start_flag = False

    def add_tensor(self, t: Union[torch.Tensor, ShardedTensor]) -> None:
        if not self._start_flag:
            return
        t_payload = t.payload if isinstance(t, ShardedTensor) else t
        mem_use = _col_tensor_mem_usage(t_payload)
        if t_payload.device.type == 'cuda':
            self._cuda_usage += mem_use
        elif t_payload.device.type == 'cpu':
            self._cpu_usage += mem_use
        else:
            raise TypeError

    def delete_tensor(self, t: Union[torch.Tensor, ShardedTensor]) -> None:
        if not self._start_flag:
            return
        t_payload = t.payload if isinstance(t, ShardedTensor) else t
        mem_use = _col_tensor_mem_usage(t_payload)
        if t_payload.device.type == 'cuda':
            self._cuda_usage -= mem_use
        elif t_payload.device.type == 'cpu':
            self._cpu_usage -= mem_use
        else:
            raise TypeError

    def clear(self) -> None:
        self._cuda_usage = 0
        self._cpu_usage = 0

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @property
    def cuda_usage(self):
        return self._cuda_usage


GLOBAL_MODEL_DATA_TRACER = ModelDataTracer()
