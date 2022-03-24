import torch
from colossalai.utils import get_current_device
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER

from typing import Union

_GLOBAL_CUDA_MEM_FRACTION = 1.0


def colo_set_process_memory_fraction(ratio: float) -> None:
    """colo_set_process_memory_fraction 

    set how much cuda memory used on the gpu belonging to the current process.

    Args:
        ratio (float): a ratio between 0. ~ 1.
    """
    global _GLOBAL_CUDA_MEM_FRACTION
    _GLOBAL_CUDA_MEM_FRACTION = ratio
    torch.cuda.set_per_process_memory_fraction(_GLOBAL_CUDA_MEM_FRACTION, get_current_device())


def colo_cuda_memory_capacity() -> float:
    """
    Get cuda memory capacity of the current cuda.
    """
    return torch.cuda.get_device_properties(get_current_device()).total_memory * _GLOBAL_CUDA_MEM_FRACTION


def colo_model_data_tensor_move(src_t: Union[ShardedTensor, torch.Tensor], tgt_t: Union[ShardedTensor,
                                                                                        torch.Tensor]) -> None:
    """ 
    A colossal API for model data tensor move. 
    The src and target tensors could be resident on both CPU and GPU.
    
    NOTE() The source tensor payload will be removed after this function.
    
    The function will record the communication volume between CPU and GPU.
    Args:
        t_src (Union[ShardedTensor, torch.Tensor]): source tensor
        tgt_t (Union[ShardedTensor, torch.Tensor]): target tensor
    """
    if isinstance(src_t, ShardedTensor):
        src_t_payload = src_t.payload
    else:
        src_t_payload = src_t.data
    src_dev = src_t_payload.device
    if isinstance(tgt_t, ShardedTensor):
        tgt_t_payload = tgt_t.payload
    else:
        tgt_t_payload = tgt_t.data
    tgt_dev = tgt_t_payload.device

    if src_dev.type == 'cuda' and tgt_dev.type == 'cpu':
        GLOBAL_MODEL_DATA_TRACER.delete_tensor(src_t_payload)
    elif src_dev.type == 'cpu' and tgt_dev.type == 'cuda':
        GLOBAL_MODEL_DATA_TRACER.add_tensor(tgt_t_payload)
    tgt_t_payload.copy_(src_t_payload)

    # remove payload of src_t
    if isinstance(src_t, ShardedTensor):
        src_t.reset_payload(torch.tensor([], device=src_dev, dtype=src_t_payload.dtype))
    else:
        src_t.data = torch.tensor([], device=src_dev, dtype=src_t_payload.dtype)


def colo_model_data_move_to_cpu(t: Union[ShardedTensor, torch.Tensor]) -> None:
    """colo_model_data_move_to_cpu 

    move a model data tensor from gpu to cpu

    Args:
        t (Union[ShardedTensor, torch.Tensor]): _description_
    """

    if isinstance(t, ShardedTensor):
        t_payload = t.payload
    elif isinstance(t, torch.Tensor):
        t_payload = t
    else:
        raise TypeError('colo_model_data_move_to_cpu dose not accept type {type(t)}')

    if t_payload.device.type == 'cpu':
        return

    # TODO() optimize the tensor moving with non-blocking
    GLOBAL_MODEL_DATA_TRACER.delete_tensor(t_payload)
    t_payload.data = t_payload.data.cpu()
