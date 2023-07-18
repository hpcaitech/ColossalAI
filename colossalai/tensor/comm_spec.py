import operator
from enum import Enum
from functools import reduce

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

__all__ = [
    'CollectiveCommPattern',
    'CommSpec',
]


def _all_gather(tensor, comm_spec):
    '''
    Implement all gather operation on device mesh based on information provided by comm_spec.
    '''
    process_groups = comm_spec.device_mesh.get_process_group_for_all_axes()
    process_group = process_groups[comm_spec.logical_process_axis]

    tensor_list = [
        torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        for _ in range(comm_spec.device_mesh.shape[comm_spec.logical_process_axis])
    ]
    # without this contiguous operation, the all gather may get some unexpected results.
    tensor = tensor.contiguous()
    dist.all_gather(tensor_list, tensor, group=process_group)
    output = torch.cat(tuple(tensor_list), comm_spec.gather_dim).contiguous()
    return output


def _split(tensor, comm_spec):
    '''
    Implement shard operation on device mesh based on information provided by comm_spec.
    '''
    process_groups = comm_spec.device_mesh.get_process_group_for_all_axes()
    process_group = process_groups[comm_spec.logical_process_axis]

    dim = comm_spec.shard_dim
    length = tensor.shape[comm_spec.shard_dim] // dist.get_world_size(process_group)
    start = length * dist.get_rank(process_group)
    output = torch.narrow(tensor, dim, start, length).contiguous()
    return output


def _all_to_all(tensor, comm_spec):
    '''
    Implement all to all operation on device mesh based on information provided by comm_spec.
    '''
    process_groups = comm_spec.device_mesh.get_process_group_for_all_axes()
    process_group = process_groups[comm_spec.logical_process_axis]
    world_size = dist.get_world_size(process_group)

    new_shape = list(tensor.shape)
    new_shape[comm_spec.shard_dim] = new_shape[comm_spec.shard_dim] // world_size
    new_shape = torch.Size(new_shape)
    output_tensor_list = [torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
    dim = comm_spec.shard_dim
    length = tensor.shape[comm_spec.shard_dim] // world_size
    input_tensor_list = [torch.narrow(tensor, dim, length * i, length).contiguous() for i in range(world_size)]
    group = process_group
    dist.all_to_all(output_tensor_list, input_tensor_list, group)
    output = torch.cat(tuple(output_tensor_list), comm_spec.gather_dim).contiguous()
    return output


def _all_reduce(tensor, comm_spec, async_op=False):
    '''
    Implement all reduce operation on device mesh based on information provided by comm_spec.
    '''
    process_groups = comm_spec.device_mesh.get_process_group_for_all_axes()
    process_group = process_groups[comm_spec.logical_process_axis]

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    dist.all_reduce(tensor, op=ReduceOp.SUM, group=process_group, async_op=async_op)
    return tensor


def _mix_gather(tensor, comm_spec):
    '''
    Implement mix gather operation on device mesh based on information provided by comm_spec.
    Mix gather is the all-gather operation on all devices in the device_mesh(FlattenDeviceMesh) of the comm_spec. It is
    different from _all_gather because _mix_gather does all-gather in two dimensions of device mesh, while _all_gather
    only does all-gather in one dimension.
    Assume index of f and b target pairs are 'f' and 'b'
    ShardingSpec => gather_dim, logical_process_axes
    S0S1 => [b, f], (1, 0)
    S1S0 => [b, f], (0, 1)
    S01R => [f], (1, 1)
    RS01 => [b], (1, 1)
    Example:
    mesh_shape = (2,4)
            # [[0, 1, 2, 3],
            #  [4, 5, 6, 7]]
            # return {0: [0, 4, 1, 5, 2, 6, 3, 7], 1: [0, 1, 2, 3, 4, 5, 6, 7]}
    S0S1:
    leading_group_dim = 1
    process_group = "[0, 1, 2, 3, 4, 5, 6, 7]"
    tensor_list = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] # [(slice_id_f, slice_id_b),...]
    mesh_shape = (2,4)
    cat_slice = [4,2]
    tmp_tensor_list = [(...,shape[f],shape[b]*4,...),(...,shape[f],shape[b]*4,...)]
    tmp_tensor_list[0] = torch.cat(((0,0),(0,1),(0,2),(0,3)), dim=b)
    tmp_tensor_list[1] = torch.cat(((1,0),(1,1),(1,2),(1,3)), dim=b)
    output = torch.cat((tmp_tensor_list[0],tmp_tensor_list[1]), dim=a)
    S1S0:
    leading_group_dim = 0
    process_group = "[0, 4, 1, 5, 2, 6, 3, 7]"
    tensor_list = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)]
    mesh_shape = (2,4)
    cat_slice = [2,4]
    tmp_tensor_list = [(...,shape[f],shape[b]*2,...),(...,shape[f],shape[b]*2,...),(...,shape[f],shape[b]*2,...),(...,shape[f],shape[b]*2,...)]
    tmp_tensor_list[0] = torch.cat(((0,0),(0,1)), dim=b)
    tmp_tensor_list[1] = torch.cat(((1,0),(1,1)), dim=b)
    tmp_tensor_list[2] = torch.cat(((2,0),(2,1)), dim=b)
    tmp_tensor_list[3] = torch.cat(((3,0),(3,1)), dim=b)
    S10R:
    leading_group_dim = 0
    process_group = "[0, 4, 1, 5, 2, 6, 3, 7]"
    tensor_list = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)]
    S01R:
    leading_group_dim = 1
    process_group = "[0, 1, 2, 3, 4, 5, 6, 7]"
    tensor_list = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)]
    '''
    total_slices = comm_spec.device_mesh.shape[0]
    tensor_list = [torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(total_slices)]
    leading_group_dim = comm_spec.logical_process_axes[0]
    assert len(comm_spec.device_mesh.process_groups_dict) == 1
    _, process_group = comm_spec.device_mesh.process_groups_dict[0][0]
    process_number_list = comm_spec.device_meshes.process_number_dict[leading_group_dim]

    # Global all_gather
    dist.all_gather(tensor_list, tensor, group=process_group)

    # This is very ugly. I'm figuring out more elegant methods
    tensor_list_sorted = [
        torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(total_slices)
    ]
    for i in range(total_slices):
        tensor_list_sorted[i] = tensor_list[process_number_list[i]]
    tensor_list = tensor_list_sorted

    if comm_spec.logical_process_axes[0] == comm_spec.logical_process_axes[1]:
        output = torch.cat(tuple(tensor_list), comm_spec.gather_dim[0]).contiguous()
    else:
        mesh_shape = comm_spec.device_meshes.shape
        cat_slice = [mesh_shape[comm_spec.logical_process_axes[0]], mesh_shape[comm_spec.logical_process_axes[1]]]
        tmp_tensor_shape = list(tensor.shape)
        tmp_tensor_shape[comm_spec.gather_dim[0]] *= cat_slice[0]
        tmp_tensor_shape = torch.Size(tmp_tensor_shape)
        tmp_tensor_list = [
            torch.zeros(tmp_tensor_shape, dtype=tensor.dtype, device=tensor.device) for _ in range(cat_slice[1])
        ]
        for i in range(cat_slice[1]):
            tmp_tensor_list[i] = torch.cat(tuple(tensor_list[i * cat_slice[0]:(i + 1) * cat_slice[0]]),
                                           comm_spec.gather_dim[0]).contiguous()
        output = torch.cat(tuple(tmp_tensor_list), comm_spec.gather_dim[1]).contiguous()

    return output


def _mix_split(tensor, comm_spec):
    '''
    Implement mix split operation. Mix split is only called for the backward of mix gather (Use ctx to keep consistent)
    Mix split shards the tensor on device mesh based on information provided by comm_spec. It is different from split
    because _mix_split shards the tensor in two dimensions of device mesh, while _split only shards in one dimension.
    Assume index of f and b target pairs are 'f' and 'b'
    S0S1 => [b, f], (1, 0)
    S1S0 => [b, f], (0, 1)
    S01R => [f], (0, 0)
    RS01 => [b], (0, 0)
    Example:
    mesh_shape = (2,4)
            # [[0, 1, 2, 3],
            #  [4, 5, 6, 7]]
            # return {0: [0, 4, 1, 5, 2, 6, 3, 7], 1: [0, 1, 2, 3, 4, 5, 6, 7]}
    '''
    mesh_shape = comm_spec.device_meshes.shape
    dim = comm_spec.gather_dim
    total_slices = comm_spec.device_mesh.shape[0]

    # Get global rank
    rank = dist.get_rank()

    leading_group_dim = comm_spec.logical_process_axes[0]
    process_number_list = comm_spec.device_meshes.process_number_dict[leading_group_dim]
    rank = process_number_list.index(rank)

    if comm_spec.logical_process_axes[0] == comm_spec.logical_process_axes[1]:
        length = tensor.shape[dim[0]] // total_slices
        start = length * rank
        output = torch.narrow(tensor, dim[0], start, length).contiguous()
    else:
        tensor_shape = [tensor.shape[dim[0]], tensor.shape[dim[1]]]
        rank_slice = [mesh_shape[comm_spec.logical_process_axes[0]], mesh_shape[comm_spec.logical_process_axes[1]]]
        length = [tensor_shape[0] // rank_slice[0], tensor_shape[1] // rank_slice[1]]
        start = [(rank % rank_slice[0]) * length[0], (rank // rank_slice[0]) * length[1]]
        tmp_output = torch.narrow(tensor, dim[0], start[0], length[0]).contiguous()
        output = torch.narrow(tmp_output, dim[1], start[1], length[1]).contiguous()

    return output


class _ReduceGrad(torch.autograd.Function):
    """
    A customized communication operation which forward is an identity operation,
    backward is all_reduce operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output, ctx.comm_spec), None


class _ReduceInput(torch.autograd.Function):
    """
    A customized communication operation which forward is all_reduce operation,
    backward is an identity operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _all_reduce(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        return _all_reduce(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    A customized communication operation which forward is split operation,
    backward is an all gather operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return _split(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_gather(grad_output, ctx.comm_spec), None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    A customized communication operation which forward is an all gather operation,
    backward is split operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _all_gather(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return _all_gather(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.comm_spec), None


class _AllToAll(torch.autograd.Function):
    """
    A customized communication operation which forward is an all to all operation,
    backward is an all to all operation.

    Args:
        input_: input matrix.
        comm_spec: comm_spec will give information like process group, rank list, etc.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _all_to_all(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        output = _all_to_all(input_, comm_spec)
        comm_spec_for_backward = CommSpec(comm_pattern=comm_spec.comm_pattern,
                                          sharding_spec=comm_spec.sharding_spec,
                                          gather_dim=comm_spec.shard_dim,
                                          shard_dim=comm_spec.gather_dim,
                                          logical_process_axis=comm_spec.logical_process_axis)
        ctx.comm_spec = comm_spec_for_backward
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return _all_to_all(grad_outputs, ctx.comm_spec), None


class _MixGatherForwardMixSplitBackward(torch.autograd.Function):

    @staticmethod
    def symbolic(graph, input_):
        return _mix_gather(input_)

    @staticmethod
    def forward(ctx, input_, comm_spec):
        ctx.comm_spec = comm_spec
        return _mix_gather(input_, comm_spec)

    @staticmethod
    def backward(ctx, grad_output):
        return _mix_split(grad_output, ctx.comm_spec), None


def reduce_grad(input_, comm_spec):
    return _ReduceGrad.apply(input_, comm_spec)


def reduce_input(input_, comm_spec):
    return _ReduceInput.apply(input_, comm_spec)


def split_forward_gather_backward(input_, comm_spec):
    return _SplitForwardGatherBackward.apply(input_, comm_spec)


def gather_forward_split_backward(input_, comm_spec):
    return _GatherForwardSplitBackward.apply(input_, comm_spec)


def all_to_all(input_, comm_spec):
    return _AllToAll.apply(input_, comm_spec)


def mixgather_forward_split_backward(input_, comm_spec):
    return _MixGatherForwardMixSplitBackward.apply(input_, comm_spec)


class CollectiveCommPattern(Enum):
    GATHER_FWD_SPLIT_BWD = 'gather_fwd_split_bwd'
    ALL2ALL_FWD_ALL2ALL_BWD = 'all2all_fwd_all2all_bwd'
    SPLIT_FWD_GATHER_BWD = 'split_fwd_gather_bwd'
    ALLREDUCE_FWD_IDENTITY_BWD = 'all_reduce_fwd_identity_bwd'
    IDENTITY_FWD_ALLREDUCE_BWD = 'identity_fwd_all_reduce_bwd'
    MIXGATHER_FWD_SPLIT_BWD = "mixgather_fwd_split_bwd"


class CommSpec:
    '''
    Communication spec is used to record the communication action. It has two main functions:
    1. Compute the communication cost which will be used in auto parallel solver.
    2. Convert the communication spec to real action which will be used in runtime.
    It contains comm_pattern to determine the
    communication method, sharding_spec to determine the communication size, gather_dim and shard_dim
    to determine the buffer shape, and logical_process_axis

    Argument:
        comm_pattern(CollectiveCommPattern): describe the communication method used in this spec.
        sharding_spec(ShardingSpec): This is sharding spec of the tensor which will join the communication action.
        gather_dim(int, Optional): The gather_dim of the tensor will be gathered.
        shard_dim(int, Optional): The shard_dim of the tensor will be sharded.
        logical_process_axis(Union(int, List[int]), Optional): The mesh_dim to implement the communication action.
    '''

    def __init__(self,
                 comm_pattern,
                 sharding_spec,
                 gather_dim=None,
                 shard_dim=None,
                 logical_process_axis=None,
                 forward_only=False,
                 mix_gather=False):
        self.comm_pattern = comm_pattern
        self.sharding_spec = sharding_spec
        self.gather_dim = gather_dim
        self.shard_dim = shard_dim
        self.logical_process_axis = logical_process_axis
        self.forward_only = forward_only
        if isinstance(self.logical_process_axis, list):
            if not mix_gather:
                self.device_mesh = self.sharding_spec.device_mesh.flatten()
                self.logical_process_axis = 0
            else:
                self.device_meshes = self.sharding_spec.device_mesh.flatten_device_meshes
                self.device_mesh = self.sharding_spec.device_mesh.flatten_device_mesh
                # Create a new member `logical_process_axes` to distinguish from original flatten
                self.logical_process_axes = logical_process_axis
        else:
            self.device_mesh = self.sharding_spec.device_mesh

    def __repr__(self):
        res_list = ["CommSpec:("]
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            res_list.append(f"comm_pattern:GATHER_FWD_SPLIT_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
            res_list.append(f"comm_pattern:ALL2ALL_FWD_ALL2ALL_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis: {self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
            res_list.append(f"comm_pattern:SPLIT_FWD_GATHER_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.shard_dim}, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
            res_list.append(f"comm_pattern:ALLREDUCE_FWD_IDENTITY_BWD, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
            res_list.append(f"comm_pattern:IDENTITY_FWD_ALLREDUCE_BWD, ")
            res_list.append(f"logical_process_axis:{self.logical_process_axis})")
        elif self.comm_pattern == CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD:
            res_list.append(f"comm_pattern:MIXGATHER_FWD_SPLIT_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"logical_process_asex:{self.logical_process_axes})")

        return ''.join(res_list)

    def get_comm_cost(self):
        '''
        For all_gather, all2all, and all_reduce operation, the formula provided in DeviceMesh with alpha-beta model is used to
        compute the communication cost.
        For shard operation, it is an on-chip operation, so the communication cost is zero.
        '''
        comm_size = reduce(operator.mul, self.sharding_spec.get_sharded_shape_per_device(), 1)
        cost_dict = {}
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            forward_communication_cost = self.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)
            # give a tiny cost to shard
            backward_communication_cost = 100

        if self.comm_pattern == CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD:
            forward_communication_cost = self.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)
            # grad should have same shape as input tensor
            # all to all operation has same logical process axis as forward.
            backward_communication_cost = self.device_mesh.all_to_all_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD:
            forward_communication_cost = self.device_mesh.all_reduce_cost(comm_size, self.logical_process_axis)
            backward_communication_cost = 0

        if self.comm_pattern == CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD:
            forward_communication_cost = 0
            backward_communication_cost = self.device_mesh.all_reduce_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.SPLIT_FWD_GATHER_BWD:
            # give a tiny cost to shard
            forward_communication_cost = 100
            backward_communication_cost = self.device_mesh.all_gather_cost(comm_size, self.logical_process_axis)

        if self.comm_pattern == CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD:
            # no need for axis because all devices are used in mix_gather
            forward_communication_cost = self.device_mesh.mix_gather_cost(comm_size)
            backward_communication_cost = 100

        if self.forward_only:
            cost_dict["forward"] = forward_communication_cost
            cost_dict["backward"] = 0
            cost_dict["total"] = cost_dict["forward"] + cost_dict["backward"]
        else:
            cost_dict["forward"] = forward_communication_cost
            cost_dict["backward"] = backward_communication_cost
            cost_dict["total"] = cost_dict["forward"] + cost_dict["backward"]

        return cost_dict

    def covert_spec_to_action(self, tensor):
        '''
        Convert CommSpec into runtime action, implement real collection communication to target tensor.
        The collection communication action is directed by the CommSpec.

        Argument:
            tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
        '''
        if self.comm_pattern in pattern_to_func_dict:
            tensor = pattern_to_func_dict[self.comm_pattern](tensor, self)
        else:
            tensor = tensor
        return tensor


pattern_to_func_dict = {
    CollectiveCommPattern.GATHER_FWD_SPLIT_BWD: gather_forward_split_backward,
    CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD: all_to_all,
    CollectiveCommPattern.SPLIT_FWD_GATHER_BWD: split_forward_gather_backward,
    CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD: reduce_input,
    CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD: reduce_grad,
    CollectiveCommPattern.MIXGATHER_FWD_SPLIT_BWD: mixgather_forward_split_backward,
}
