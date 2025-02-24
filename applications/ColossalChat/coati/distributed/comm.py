from typing import Any, Dict

import ray.util.collective as cc
import torch
import torch.distributed.distributed_c10d as c10d
from packaging.version import Version


def ray_broadcast_object(obj: Any, src: int = 0, device=None, group_name: str = "default") -> Any:
    rank = cc.get_rank(group_name)
    if rank == src:
        if Version(torch.__version__) >= Version("2.3.0"):
            obj_tensor, size_tensor = c10d._object_to_tensor(obj, device=device, group=None)
        elif Version(torch.__version__) >= Version("1.13.0"):
            obj_tensor, size_tensor = c10d._object_to_tensor(obj, device=device)
        else:
            obj_tensor, size_tensor = c10d._object_to_tensor(obj)
        obj_tensor = obj_tensor.to(device)
        size_tensor = size_tensor.to(device)
    else:
        size_tensor = torch.empty(1, dtype=torch.int64, device=device)
    cc.broadcast(size_tensor, src, group_name)
    if rank != src:
        obj_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device=device)
    cc.broadcast(obj_tensor, src, group_name)
    if rank != src:
        if Version(torch.__version__) >= Version("2.3.0"):
            obj = c10d._tensor_to_object(obj_tensor, size_tensor.item(), group=None)
        else:
            obj = c10d._tensor_to_object(obj, size_tensor.item())
    return obj


def ray_broadcast_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor], src: int = 0, device=None, group_name: str = "default"
) -> Dict[str, torch.Tensor]:
    rank = cc.get_rank(group_name)
    if rank == src:
        metadata = []
        non_tensor_dict = {}
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                metadata.append((k, v.shape, v.dtype))
            else:
                non_tensor_dict[k] = v
    else:
        metadata = None
        non_tensor_dict = None

    data_to_broadcast = (metadata, non_tensor_dict)
    data_to_broadcast = ray_broadcast_object(data_to_broadcast, src, device, group_name)
    metadata, non_tensor_dict = data_to_broadcast

    if rank != src:
        out_dict = non_tensor_dict
    for k, shape, dtype in metadata:
        if rank == src:
            tensor = tensor_dict[k]
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        cc.broadcast(tensor, src, group_name)
        if rank != src:
            out_dict[k] = tensor
    if rank == src:
        out_dict = tensor_dict
    return out_dict
