from typing import Tuple, Union

import torch

from .stateful_tensor import StatefulTensor


def is_storage_empty(tensor: torch.Tensor) -> bool:
    return tensor.storage().size() == 0


def free_storage(tensor: torch.Tensor) -> None:
    if not is_storage_empty(tensor):
        tensor.storage().resize_(0)


def alloc_storage(tensor: torch.Tensor) -> None:
    if is_storage_empty(tensor):
        tensor.storage().resize_(tensor.numel())


def colo_tensor_mem_usage(tensor: Union[torch.Tensor, StatefulTensor]) -> Tuple[int, int]:
    if isinstance(tensor, StatefulTensor):
        t = tensor.payload
    elif isinstance(tensor, torch.Tensor):
        t = tensor
    else:
        return 0, 0

    cuda_use, cpu_use = 0, 0

    mem_use = t.storage().size() * t.element_size()
    if t.device.type == "cuda":
        cuda_use += mem_use
    elif t.device.type == "cpu":
        cpu_use += mem_use

    return cuda_use, cpu_use


def colo_model_data_tensor_move(
    src_t: Union[StatefulTensor, torch.Tensor], tgt_t: Union[StatefulTensor, torch.Tensor]
) -> None:
    """
    A colossal API for model data tensor move.
    The src and target tensors could be resident on both CPU and GPU.

    NOTE() The source tensor payload will be removed after this function.

    The function will record the communication volume between CPU and GPU.
    Args:
        src_t (Union[StatefulTensor, torch.Tensor]): source tensor
        tgt_t (Union[StatefulTensor, torch.Tensor]): target tensor
    """
    if isinstance(src_t, StatefulTensor):
        src_t_payload = src_t.payload
    else:
        src_t_payload = src_t.data
    src_dev = src_t_payload.device

    if isinstance(tgt_t, StatefulTensor):
        tgt_t_payload = tgt_t.payload
    else:
        tgt_t_payload = tgt_t.data

    tgt_t_payload.copy_(src_t_payload)

    # remove payload of src_t
    if isinstance(src_t, StatefulTensor):
        src_t.set_null()
    else:
        src_t.data = torch.empty(0, device=src_dev, dtype=src_t_payload.dtype)


def colo_model_data_tensor_move_inline(
    t: Union[StatefulTensor, torch.Tensor], target_device: Union[torch.device, int]
) -> None:
    """
    move a tensor to the target_device
    Args:
        t (Union[StatefulTensor, torch.Tensor]): the tensor be moved
        target_device: a target device, if type is int, it the index of cuda card.
    """
    if not isinstance(target_device, torch.device):
        target_device = torch.device(f"cuda:{target_device}")

    if isinstance(t, torch.Tensor):
        t.data = t.data.to(target_device)
    elif isinstance(t, StatefulTensor):
        t.move_to(target_device)
    else:
        raise TypeError(f"colo_model_data_tensor_move_inline dose not accept type {type(t)}")


def colo_model_data_move_to_cpu(t: Union[StatefulTensor, torch.Tensor]) -> None:
    """colo_model_data_move_to_cpu
    move a model data tensor from gpu to cpu
    Args:
        t (Union[StatefulTensor, torch.Tensor]): _description_
    """
    # TODO() optimize the tensor moving with non-blocking
    if isinstance(t, torch.Tensor):
        t.data = t.data.cpu()
    elif isinstance(t, StatefulTensor):
        t.move_to(torch.device("cpu"))
    else:
        raise TypeError(f"colo_model_data_move_to_cpu dose not accept type {type(t)}")


def colo_model_tensor_clone(t: Union[StatefulTensor, torch.Tensor], target_device: torch.device) -> torch.Tensor:
    """
    Clone a model data tensor
    Args:
        t (Union[StatefulTensor, torch.Tensor]): a model data tensor
        target_device (torch.device): the target device
    Returns:
        torch.Tensor: a cloned torch tensor
    """
    # TODO() rename this function
    colo_model_data_tensor_move_inline(t, target_device)
    t_payload = t.payload if isinstance(t, StatefulTensor) else t
    return t_payload
