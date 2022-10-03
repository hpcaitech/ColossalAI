import torch
from enum import Enum, auto
from typing import List
from colossalai.tensor.sharding_spec import ShardingSpec

__all__ = ['BroadcastType', 'is_broadcastable', 'get_broadcast_shape', 'recover_sharding_spec_for_broadcast_shape']


class BroadcastType(Enum):
    EQUAL = auto()
    PADDDING = auto()
    MULTIPLE = auto()


def is_broadcastable(shape1: torch.Size, shape2: torch.Size) -> bool:
    """
    Check if two shapes are broadcastable to each other.
    """
    for s1, s2 in zip(shape1[::-1], shape2[::-1]):
        if s1 == 1 or s2 == 1 or s1 == s2:
            pass
        else:
            return False
    return True


def get_broadcast_shape(shape1: torch.Size, shape2: torch.Size) -> List[int]:
    """
    Compute the broadcast shape given two shapes.
    """
    assert is_broadcastable(shape1, shape2), f'{shape1} and {shape2} are not broadcastable'
    shape1_reverse = shape1[::-1]
    shape2_reverse = shape2[::-1]
    min_common_dim = min(len(shape1), len(shape2))
    dims = []
    for s1, s2 in zip(shape1_reverse, shape2_reverse):
        dims.append(max(s1, s2))

    # append the remaining dims
    dims.extend(shape1_reverse[min_common_dim:])
    dims.extend(shape2_reverse[min_common_dim:])
    return dims[::-1]


def recover_sharding_spec_for_broadcast_shape(logical_sharding_spec: ShardingSpec, logical_shape: torch.Size,
                                              physical_shape: torch.Size) -> ShardingSpec:
    """
    This function computes the sharding spec for the physical shape of a broadcast tensor.

    Args:
        logical_sharding_spec (ShardingSpec): the sharding spec for the broadcast tensor
        logical_shape (torch.Size): logical shape is the broadcast shape of a tensor
        physical_shape (torch.Size): the shape of the tensor before broadcasting
    """
    # get the number of dimensions
    logical_num_dims = len(logical_shape)
    physical_num_dims = len(physical_shape)

    # track the dim and its broadcasting type
    logical_dim_broadcast_info = {}

    for i in range(logical_num_dims):
        # get the trailing dim size
        logical_dim_idx = logical_num_dims - i - 1
        phyiscal_dim_idx = physical_num_dims - i - 1
        logical_dim_size = logical_shape[logical_dim_idx]

        if phyiscal_dim_idx >= 0:
            physical_dim_size = physical_shape[phyiscal_dim_idx]

            if physical_dim_size == logical_dim_size:
                logical_dim_broadcast_info[logical_dim_idx] = BroadcastType.EQUAL
            elif physical_dim_size == 1 and physical_dim_size != logical_dim_size:
                logical_dim_broadcast_info[logical_dim_idx] = BroadcastType.MULTIPLE
        else:
            logical_dim_broadcast_info[logical_dim_idx] = BroadcastType.PADDDING

    # generate the sharding spec for the physical shape
    physical_dim_partition = {}
    logical_dim_partition = logical_sharding_spec.dim_partition_dict

    for shape_dim, mesh_dim in logical_dim_partition.items():
        logical_broadcast_type = logical_dim_broadcast_info[shape_dim]

        if logical_broadcast_type == BroadcastType.PADDDING or logical_broadcast_type == BroadcastType.MULTIPLE:
            pass
        else:
            # get the corresponding physical dim
            physical_dim = physical_num_dims - (logical_num_dims - shape_dim)
            physical_dim_partition[physical_dim] = mesh_dim

    physical_sharding_spec = ShardingSpec(device_mesh=logical_sharding_spec.device_mesh,
                                          entire_shape=physical_shape,
                                          dim_partition_dict=physical_dim_partition)

    return physical_sharding_spec
