from typing import Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from colossalai.device.device_mesh import DeviceMesh

from .d_tensor import DTensor
from .sharding_spec import ShardingSpec


def shard_rowwise(tensor: torch.Tensor, group_or_device_mesh: Union[ProcessGroup, DeviceMesh] = None) -> DTensor:
    """
    Shard the first dim of the given tensor
    """
    # if the group_or_device_mesh is None, we shard the tensor with respect to the global process group
    if group_or_device_mesh is None:
        group_or_device_mesh = dist.GroupMember.WORLD

    if isinstance(group_or_device_mesh, ProcessGroup):
        device_mesh = DeviceMesh.from_process_group(group_or_device_mesh)
    else:
        assert len(group_or_device_mesh.shape) == 1, 'Only 1D DeviceMesh is accepted for row-wise sharding.'
        device_mesh = group_or_device_mesh
    sharding_spec = ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={0: [0]})
    return DTensor(tensor, device_mesh, sharding_spec)


def shard_colwise(tensor: torch.Tensor, group_or_device_mesh: Union[ProcessGroup, DeviceMesh] = None) -> DTensor:
    """
    Shard the first dim of the given tensor
    """
    # if the group_or_device_mesh is None, we shard the tensor with respect to the global process group
    if group_or_device_mesh is None:
        group_or_device_mesh = dist.GroupMember.WORLD

    if isinstance(group_or_device_mesh, ProcessGroup):
        device_mesh = DeviceMesh.from_process_group(group_or_device_mesh)
    else:
        assert len(group_or_device_mesh.shape) == 1, 'Only 1D DeviceMesh is accepted for row-wise sharding.'
        device_mesh = group_or_device_mesh
    sharding_spec = ShardingSpec(dim_size=tensor.dim(), dim_partition_dict={-1: [0]})
    return DTensor(tensor, device_mesh, sharding_spec)
