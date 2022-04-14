from colossalai.context.singleton_meta import SingletonMeta
import torch
from typing import Tuple, Optional
from colossalai.logging import DistributedLogger


def colo_model_optimizer_usage(optim) -> Tuple[int, int]:
    """Trace the optimizer memory usage

    Args:
        optim (ShardedOptimV2): an instance of ShardedOptimver

    Returns:
        Tuple[int, int]: cuda/cpu memory usage in Byte
    """
    if optim is None:
        return 0, 0
    assert hasattr(optim, 'get_memory_usage'), f"{type(optim)} has no attr get_memory_usage()"
    return optim.get_memory_usage()


def colo_model_mem_usage(model: torch.nn.Module) -> Tuple[int, int]:
    """ 
    Trace the model memory usage.
    Args:
        model (torch.nn.Module): a torch model

    Returns:
        Tuple[int, int]: cuda memory usage in Byte, cpu memory usage in Byte
    """
    if model is None:
        return 0, 0

    def _get_tensor_mem_use(t: Optional[torch.Tensor]):
        if t is None:
            return 0, 0
        assert isinstance(t, torch.Tensor)
        _cpu_mem_usage, _cuda_mem_usage = 0, 0
        if t.device.type == 'cpu':
            _cpu_mem_usage += t.numel() * t.element_size()
        elif t.device.type == 'cuda':
            _cuda_mem_usage += t.numel() * t.element_size()
        return _cuda_mem_usage, _cpu_mem_usage

    cuda_mem_usage = 0
    cpu_mem_usage = 0
    for param in model.parameters():
        if hasattr(param, 'colo_attr'):
            t_cuda, t_cpu = param.colo_attr.get_memory_usage()
            cuda_mem_usage += t_cuda
            cpu_mem_usage += t_cpu
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
        self._opitimizer = None

    def _get_mem_usage(self) -> Tuple[int, int]:
        """
        get the memory usage of the model registered.
        Returns:
            Tuple[int, int]: cuda, cpu mem usage
        """
        cuda_use_opt, cpu_use_opt = colo_model_optimizer_usage(self._opitimizer)
        cuda_use_model, cpu_use_model = colo_model_mem_usage(self._model)
        return cuda_use_opt + cuda_use_model, cpu_use_opt + cpu_use_model

    def register_model(self, model) -> None:
        if self._model is not None:
            self._logger.warning("ModelDataTracer has already registered a model")
        self._model = model

    def register_optimizer(self, optimizer) -> None:
        if self._opitimizer is not None:
            self._logger.warning("ModelDataTracer has already registered an optimizer")
        self._opitimizer = optimizer

    @property
    def cpu_usage(self):
        _, cpu_usage = self._get_mem_usage()
        return cpu_usage

    @property
    def cuda_usage(self):
        cuda_usage, _ = self._get_mem_usage()
        return cuda_usage

    @property
    def both_mem_usage(self):
        return self._get_mem_usage()


GLOBAL_MODEL_DATA_TRACER = ModelDataTracer()
