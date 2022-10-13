import torch
from typing import Dict
from colossalai.tensor.sharding_spec import ShardingSpec
from copy import deepcopy


def switch_partition_dim(sharding_spec: ShardingSpec, dim1: int, dim2: int) -> ShardingSpec:
    """
    Switch the sharding mesh dimensions for two tensor dimensions. This operation is in-place.

    Args:
        sharding_spec (ShardingSpec): the sharding spec for which partition dim are switched
        dim1 (int): the tensor dimension to switch
        dim2 (int): the tensor dimension to switch
    """
    assert len(sharding_spec.entire_shape) == 2
    dim_partition_dict = sharding_spec.dim_partition_dict
    dim1_partition = dim_partition_dict.pop(dim1, None)
    dim2_partition = dim_partition_dict.pop(dim2, None)

    if dim1_partition:
        dim_partition_dict[dim2] = dim1_partition

    if dim2_partition:
        dim_partition_dict[dim1] = dim2_partition

    # re-init the sharding spec
    sharding_spec.__init__(sharding_spec.device_mesh, sharding_spec.entire_shape, dim_partition_dict)
    return sharding_spec


def update_partition_dim(sharding_spec: ShardingSpec,
                         dim_mapping: Dict[int, int],
                         physical_shape: torch.Size,
                         inplace: bool = False):
    """
    This method is used to update the partition dim dict from the logical one to the physical one.

    Args:
        sharding_spec (ShardingSpec): the sharding spec for which partition dims are updated
        dim_mapping (Dict[int, int]): the mapping from the logical tensor dimension to the physical tensor dimension
        physical_shape (torch.Size): the physical shape for the tensor
    """

    if inplace:
        current_sharding_spec = sharding_spec
    else:
        current_sharding_spec = deepcopy(sharding_spec)

    old_dim_partition_dict = current_sharding_spec.dim_partition_dict
    new_dim_partition_dict = {}

    # assign new dim
    for old_dim, new_dim in dim_mapping.items():
        mesh_dims = old_dim_partition_dict.pop(old_dim)
        new_dim_partition_dict[new_dim] = mesh_dims

    for tensor_dim, mesh_dims in old_dim_partition_dict.items():
        if tensor_dim in new_dim_partition_dict:
            raise KeyError(f"There are duplicated entries for the tensor sharding dimension {tensor_dim}")
        else:
            new_dim_partition_dict[tensor_dim] = mesh_dims

    # update sharding spec
    current_sharding_spec.__init__(device_mesh=sharding_spec.device_mesh,
                                   entire_shape=physical_shape,
                                   dim_partition_dict=new_dim_partition_dict)
    return current_sharding_spec
