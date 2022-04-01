import torch
from colossalai.zero.sharded_param.tensorful_state import StatefulTensor
from typing import Union, Tuple


def colo_tensor_mem_usage(tensor: Union[torch.Tensor, StatefulTensor]) -> Tuple[int, int]:
    if issubclass(type(tensor), StatefulTensor):
        t = tensor.payload
    elif isinstance(tensor, torch.Tensor):
        t = tensor
    else:
        return 0, 0

    cuda_use, cpu_use = 0, 0

    mem_use = t.numel() * t.element_size()
    if t.device.type == 'cuda':
        cuda_use += mem_use
    elif t.device.type == 'cpu':
        cpu_use += mem_use

    return cuda_use, cpu_use


def colo_model_data_tensor_move(src_t: Union[StatefulTensor, torch.Tensor], tgt_t: Union[StatefulTensor,
                                                                                         torch.Tensor]) -> None:
    """ 
    A colossal API for model data tensor move. 
    The src and target tensors could be resident on both CPU and GPU.
    
    NOTE() The source tensor payload will be removed after this function.
    
    The function will record the communication volume between CPU and GPU.
    Args:
        t_src (Union[StatefulTensor, torch.Tensor]): source tensor
        tgt_t (Union[StatefulTensor, torch.Tensor]): target tensor
    """
    if issubclass(type(src_t), StatefulTensor):
        src_t_payload = src_t.payload
    else:
        src_t_payload = src_t.data
    src_dev = src_t_payload.device
    if issubclass(type(tgt_t), StatefulTensor):
        tgt_t_payload = tgt_t.payload
    else:
        tgt_t_payload = tgt_t.data

    tgt_t_payload.copy_(src_t_payload)

    # remove payload of src_t
    if issubclass(type(src_t), StatefulTensor):
        src_t.reset_payload(torch.tensor([], device=src_dev, dtype=src_t_payload.dtype))
    else:
        src_t.data = torch.tensor([], device=src_dev, dtype=src_t_payload.dtype)


def colo_model_data_tensor_move_inline(t: Union[StatefulTensor, torch.Tensor], target_device: Union[torch.device,
                                                                                                    int]) -> None:
    """ 
    move a tensor to the target_device
    Args:
        t (Union[StatefulTensor, torch.Tensor]): the tensor be moved
        target_device: a traget device, if type is int, it the index of cuda card.
    """
    if isinstance(t, torch.Tensor):
        t_payload = t
    elif issubclass(type(t), StatefulTensor):
        t_payload = t.payload
    else:
        raise TypeError('colo_model_data_move_to_cpu dose not accept type {type(t)}')

    if not isinstance(target_device, torch.device):
        target_device = torch.device(f'cuda:{target_device}')

    # deal with torch.device('cpu') and torch.device('cpu:0)
    if t_payload.device.type == target_device.type:
        return
    t_payload.data = t_payload.data.to(target_device)


def colo_model_data_move_to_cpu(t: Union[StatefulTensor, torch.Tensor]) -> None:
    """colo_model_data_move_to_cpu 

    move a model data tensor from gpu to cpu

    Args:
        t (Union[StatefulTensor, torch.Tensor]): _description_
    """

    if issubclass(type(t), StatefulTensor):
        t_payload = t.payload
    elif isinstance(t, torch.Tensor):
        t_payload = t
    else:
        raise TypeError('colo_model_data_move_to_cpu dose not accept type {type(t)}')

    if t_payload.device.type == 'cpu':
        return

    # TODO() optimize the tensor moving with non-blocking
    t_payload.data = t_payload.data.cpu()


def colo_model_tensor_clone(t: Union[StatefulTensor, torch.Tensor], target_device: torch.device) -> torch.Tensor:
    """
    Clone a model data tensor

    Args:
        t (Union[StatefulTensor, torch.Tensor]): a model data tensor
        target_device (torch.device): the target device
    Returns:
        torch.Tensor: a cloned torch tensor
    """
    t_payload = t.payload if issubclass(type(t), StatefulTensor) else t

    ret = t_payload.to(target_device)
    return ret
