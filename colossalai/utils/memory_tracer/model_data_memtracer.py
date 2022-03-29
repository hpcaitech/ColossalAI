from colossalai.context.singleton_meta import SingletonMeta
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from colossalai.utils.memory_utils.utils import colo_tensor_mem_usage
import torch
from typing import Union, Tuple, Optional
from colossalai.logging import DistributedLogger


def col_model_data_mem_usage(model: torch.nn.Module) -> Tuple[int, int]:
    """ 
    Trace the model memory usage.
    Args:
        model (torch.nn.Module): a torch model

    Returns:
        Tuple[int, int]: cuda memory usage in Byte, cpu memory usage in Byte
    """

    def _get_tensor_mem_use(t: Optional[torch.Tensor]):
        if t is None:
            return
        assert isinstance(t, torch.Tensor)
        _cpu_mem_usage, _cuda_mem_usage = 0, 0
        if t.device.type == 'cpu':
            _cpu_mem_usage += t.numel() * t.element_size()
        elif t.device.type == 'cuda':
            _cuda_mem_usages += t.numel() * t.element_size()
        return _cuda_mem_usage, _cpu_mem_usage

    cuda_mem_usage = 0
    cpu_mem_usage = 0
    for param in model.parameters():
        if hasattr(param, 'col_attr'):
            para_cuda, param_cpu = param.col_attr.get_memory_usage()
            cuda_mem_usage += para_cuda
            cpu_mem_usage += param_cpu
        else:
            t_cuda, t_cpu = _get_tensor_mem_use(param.data)
            cuda_mem_usage += t_cuda
            cpu_mem_usage += t_cpu
            t_cuda, t_cpu = _get_tensor_mem_use(param.grad)
            cuda_mem_usage += t_cuda
            cpu_mem_usage += t_cpu

    return cuda_mem_usage, cpu_mem_usage


class ModelDataTracer(metaclass=SingletonMeta):
    """
    A tracer singleton to trace model data usage during runtime.
    You have to register a model on the singleton first.
    """

    def __init__(self) -> None:
        self._logger = DistributedLogger("ModelDataTracer")
        self._model = None

    def _get_mem_usage(self) -> Tuple[int, int]:
        """
        get the memory usage of the model registered.
        Returns:
            Tuple[int, int]: cuda, cpu mem usage
        """
        if self._model is None:
            self._logger.warning("The Global ModelDataTracer is using, but no model is registered on it.")
            return 0, 0
        return col_model_data_mem_usage(self._model)

    def register_model(self, model) -> None:
        self._model = model

    @property
    def cpu_usage(self):
        _, cpu_usage = self._get_mem_usage()
        return cpu_usage

    @property
    def cuda_usage(self):
        cuda_usage, _ = self._get_mem_usage()
        return cuda_usage


GLOBAL_MODEL_DATA_TRACER = ModelDataTracer()
