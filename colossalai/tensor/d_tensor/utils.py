import operator
from functools import reduce
from typing import Dict

from colossalai.tensor.d_tensor.comm_spec import CollectiveCommPattern, CommSpec
from colossalai.tensor.d_tensor.layout import Layout


def get_comm_cost(layout: Layout, comm_spec: CommSpec, forward_only: bool = False) -> Dict[str, float]:
    """
    This method is used to compute the communication cost for a given layout and comm_spec.

    For all_gather, all2all, and all_reduce operation, the formula provided in DeviceMesh with alpha-beta model is used to
    compute the communication cost. For shard operation, it is an on-chip operation, so the communication cost is a tiny cost.

    Args:
        layout: the layout of the tensor.
        comm_spec: the comm_spec to instruct the communication operation.
        forward_only: if it is True, we will just count the forward communication cost.
            If it is False, we will count both forward and backward communication cost.
    """
    comm_size = reduce(operator.mul, layout.get_sharded_shape_per_device(), 1)
    device_mesh = layout.device_mesh
    comm_pattern = comm_spec.comm_pattern
    logical_process_axis = comm_spec.logical_process_axis
    cost_dict = {}

    if comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
        # the comm size for all gather is the size of the gathered tensor
        gather_dim = comm_spec.gather_dim
        all_gather_axis = layout.sharding_spec.dim_partition_dict[gather_dim][-1]
        all_gather_size = device_mesh.shape[all_gather_axis]
        comm_size_for_all_gather = comm_size * all_gather_size
        forward_communication_cost = device_mesh.all_gather_cost(comm_size_for_all_gather, logical_process_axis)
        # give a tiny cost to shard
        backward_communication_cost = 100

    if comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
        forward_communication_cost = device_mesh.all_to_all_cost(comm_size, logical_process_axis)
        # grad should have same shape as input tensor
        # all to all operation has same logical process axis as forward.
        backward_communication_cost = device_mesh.all_to_all_cost(comm_size, logical_process_axis)

    if comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
        forward_communication_cost = device_mesh.all_reduce_cost(comm_size, logical_process_axis)
        backward_communication_cost = 0

    if comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
        forward_communication_cost = 0
        backward_communication_cost = device_mesh.all_reduce_cost(comm_size, logical_process_axis)

    if comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
        # give a tiny cost to shard
        forward_communication_cost = 100
        backward_communication_cost = device_mesh.all_gather_cost(comm_size, logical_process_axis)

    if forward_only:
        cost_dict["forward"] = forward_communication_cost
        cost_dict["backward"] = 0
        cost_dict["total"] = cost_dict["forward"] + cost_dict["backward"]
    else:
        cost_dict["forward"] = forward_communication_cost
        cost_dict["backward"] = backward_communication_cost
        cost_dict["total"] = cost_dict["forward"] + cost_dict["backward"]

    return cost_dict
