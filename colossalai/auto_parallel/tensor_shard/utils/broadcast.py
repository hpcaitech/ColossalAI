from enum import Enum, auto
from typing import List

import torch
from torch.fx.node import Node

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    CommAction,
    CommType,
    OperationData,
    OperationDataType,
)
from colossalai.tensor.comm_spec import CollectiveCommPattern, CommSpec
from colossalai.tensor.sharding_spec import ShardingSpec

__all__ = [
    "BroadcastType",
    "is_broadcastable",
    "get_broadcast_shape",
    "recover_sharding_spec_for_broadcast_shape",
    "comm_actions_for_oprands",
]


class BroadcastType(Enum):
    EQUAL = auto()
    PADDING = auto()
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
    assert is_broadcastable(shape1, shape2), f"{shape1} and {shape2} are not broadcastable"
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


def get_broadcast_dim_info(logical_shape, physical_shape):
    # get the number of dimensions
    logical_num_dims = len(logical_shape)
    physical_num_dims = len(physical_shape)

    assert (
        logical_num_dims >= physical_num_dims
    ), "The number of dimensions in the logical shape is smaller than that of the physical shape, this tensor is not broadcast!"

    # track the dim and its broadcasting type
    logical_dim_broadcast_info = {}

    for i in range(logical_num_dims):
        # get the trailing dim size
        logical_dim_idx = logical_num_dims - i - 1
        physical_dim_idx = physical_num_dims - i - 1
        logical_dim_size = logical_shape[logical_dim_idx]

        if physical_dim_idx >= 0:
            physical_dim_size = physical_shape[physical_dim_idx]

            if physical_dim_size == logical_dim_size:
                logical_dim_broadcast_info[logical_dim_idx] = BroadcastType.EQUAL
            elif physical_dim_size == 1 and physical_dim_size != logical_dim_size:
                logical_dim_broadcast_info[logical_dim_idx] = BroadcastType.MULTIPLE
        else:
            logical_dim_broadcast_info[logical_dim_idx] = BroadcastType.PADDING

    return logical_dim_broadcast_info


def recover_sharding_spec_for_broadcast_shape(
    logical_sharding_spec: ShardingSpec, logical_shape: torch.Size, physical_shape: torch.Size
) -> ShardingSpec:
    """
    This function computes the sharding spec for the physical shape of a broadcast tensor.

    Args:
        logical_sharding_spec (ShardingSpec): the sharding spec for the broadcast tensor
        logical_shape (torch.Size): logical shape is the broadcast shape of a tensor
        physical_shape (torch.Size): the shape of the tensor before broadcasting
    """
    # if the two shapes are the same, no broadcast occurs
    # we directly return the current sharding spec

    # recording the sharding dimensions removed during logical shape converting to physical one
    removed_dims = []
    if list(logical_shape) == list(physical_shape):
        return logical_sharding_spec, removed_dims

    # get the number of dimensions
    logical_num_dims = len(logical_shape)
    physical_num_dims = len(physical_shape)

    # get the broadcast info
    logical_dim_broadcast_info = get_broadcast_dim_info(logical_shape, physical_shape)

    # generate the sharding spec for the physical shape
    physical_dim_partition = {}
    logical_dim_partition = logical_sharding_spec.dim_partition_dict

    for shape_dim, mesh_dim in logical_dim_partition.items():
        logical_broadcast_type = logical_dim_broadcast_info[shape_dim]

        if logical_broadcast_type == BroadcastType.PADDING or logical_broadcast_type == BroadcastType.MULTIPLE:
            removed_dims.extend(mesh_dim)
        else:
            # get the corresponding physical dim
            physical_dim = physical_num_dims - (logical_num_dims - shape_dim)
            physical_dim_partition[physical_dim] = mesh_dim

    physical_sharding_spec = ShardingSpec(
        device_mesh=logical_sharding_spec.device_mesh,
        entire_shape=physical_shape,
        dim_partition_dict=physical_dim_partition,
    )

    return physical_sharding_spec, removed_dims


def comm_actions_for_oprands(
    node: Node, removed_dims: List[int], op_data: OperationData, sharding_spec: ShardingSpec
) -> CommAction:
    """
    This method is used to generate communication actions for oprands which lose information
    during convert logical shape to physical shape.
    """
    if len(removed_dims) == 1:
        # if list length is 1, extract element from list to avoid using flatten device mesh
        removed_dims = removed_dims[0]
    comm_spec = CommSpec(
        comm_pattern=CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD,
        sharding_spec=sharding_spec,
        logical_process_axis=removed_dims,
    )
    if op_data.type == OperationDataType.PARAM:
        comm_type = CommType.HOOK
    else:
        comm_type = CommType.BEFORE
    arg_index = -1
    for index, arg in enumerate(node.args):
        if op_data.name == str(arg):
            arg_index = index
    assert arg_index >= 0, f"op_data should be an argument of node."
    comm_action = CommAction(
        comm_spec=comm_spec,
        comm_type=comm_type,
        arg_index=arg_index,
    )
    return comm_action
