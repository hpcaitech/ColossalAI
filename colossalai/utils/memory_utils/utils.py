import torch
from colossalai.utils import get_current_device
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from colossalai.utils.memory_tracer.model_data_memtracer import GLOBAL_MODEL_DATA_TRACER

from typing import Union


def colo_cuda_memory_capacity():
    """
    Get cuda memory capacity of the current cuda.
    """
    return torch.cuda.get_device_properties(get_current_device()).total_memory


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


def colo_model_data_move_to_cpu(t: torch.Tensor):
    assert isinstance(t, torch.Tensor)
    if t.device.type == 'cpu':
        return

    GLOBAL_MODEL_DATA_TRACER.delete_tensor(t)
    t.data = t.data.cpu()
