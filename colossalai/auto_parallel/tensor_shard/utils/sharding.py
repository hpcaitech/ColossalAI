import operator
from copy import deepcopy
from functools import reduce
from typing import Dict

import torch

from colossalai.tensor.sharding_spec import ShardingSpec

__all__ = [
    "transpose_partition_dim",
    "update_partition_dim",
    "enumerate_all_possible_1d_sharding",
    "enumerate_all_possible_2d_sharding",
    "generate_sharding_size",
]


def transpose_partition_dim(sharding_spec: ShardingSpec, dim1: int, dim2: int) -> ShardingSpec:
    """
    Switch the sharding mesh dimensions for two tensor dimensions. This operation is in-place.

    Args:
        sharding_spec (ShardingSpec): the sharding spec for which partition dim are switched
        dim1 (int): the tensor dimension to switch
        dim2 (int): the tensor dimension to switch
    """
    assert len(sharding_spec.entire_shape) >= 2, "The entire_shape of the sharding spec must have at least 2 dimensions"
    dim_partition_dict = sharding_spec.dim_partition_dict

    # transpose the dim partition
    dim1_partition = dim_partition_dict.pop(dim1, None)
    dim2_partition = dim_partition_dict.pop(dim2, None)

    if dim1_partition:
        dim_partition_dict[dim2] = dim1_partition
    if dim2_partition:
        dim_partition_dict[dim1] = dim2_partition

    # get the transposed shape
    new_shape = list(sharding_spec.entire_shape[:])
    new_shape[dim2], new_shape[dim1] = new_shape[dim1], new_shape[dim2]
    new_shape = torch.Size(new_shape)

    # re-init the sharding spec
    sharding_spec.__init__(sharding_spec.device_mesh, new_shape, dim_partition_dict)
    return sharding_spec


def update_partition_dim(
    sharding_spec: ShardingSpec, dim_mapping: Dict[int, int], physical_shape: torch.Size, inplace: bool = False
):
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
    current_sharding_spec.__init__(
        device_mesh=sharding_spec.device_mesh, entire_shape=physical_shape, dim_partition_dict=new_dim_partition_dict
    )
    return current_sharding_spec


def enumerate_all_possible_2d_sharding(mesh_dim_0, mesh_dim_1, dim_size):
    dim_partition_list = []
    # enumerate all the 2D sharding cases
    for i in range(dim_size):
        for j in range(i + 1, dim_size):
            dim_partition_dict_0 = {i: [mesh_dim_0], j: [mesh_dim_1]}
            dim_partition_dict_1 = {i: [mesh_dim_1], j: [mesh_dim_0]}
            dim_partition_list.append(dim_partition_dict_0)
            dim_partition_list.append(dim_partition_dict_1)
    for i in range(dim_size):
        dim_partition_dict_flatten = {i: [mesh_dim_0, mesh_dim_1]}
        dim_partition_list.append(dim_partition_dict_flatten)

    return dim_partition_list


def enumerate_all_possible_1d_sharding(mesh_dim_0, dim_size):
    dim_partition_list = []
    # enumerate all the 1D sharding cases
    for i in range(dim_size):
        dim_partition_dict_0 = {i: [mesh_dim_0]}
        dim_partition_list.append(dim_partition_dict_0)

    return dim_partition_list


def generate_sharding_size(dim_partition_dict, device_mesh):
    total_sharding_size = 1
    for mesh_dim_list in dim_partition_dict.values():
        mesh_dim_sharding_size = [device_mesh.shape[mesh_dim] for mesh_dim in mesh_dim_list]
        sharding_size = reduce(operator.mul, mesh_dim_sharding_size)
        total_sharding_size *= sharding_size

    return total_sharding_size
