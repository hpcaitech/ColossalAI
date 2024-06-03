from enum import Enum
from typing import Dict

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

__all__ = [
    "CollectiveCommPattern",
    "CommSpec",
]


class CollectiveCommPattern(Enum):
    GATHER_FWD_SPLIT_BWD = "gather_fwd_split_bwd"
    ALL2ALL_FWD_ALL2ALL_BWD = "all2all_fwd_all2all_bwd"
    SPLIT_FWD_GATHER_BWD = "split_fwd_gather_bwd"
    ALLREDUCE_FWD_IDENTITY_BWD = "all_reduce_fwd_identity_bwd"
    IDENTITY_FWD_ALLREDUCE_BWD = "identity_fwd_all_reduce_bwd"
    MIXGATHER_FWD_SPLIT_BWD = "mixgather_fwd_split_bwd"


class CommSpec:
    """
    Communication spec is used to record the communication action. It converts the communication spec
    to real action which will be used in runtime. It contains comm_pattern to determine the
    communication method, process_group_dict to determine the process groups, gather_dim and shard_dim
    to determine the buffer shape, and logical_process_axis

    Argument:
        comm_pattern(CollectiveCommPattern): describe the communication method used in this spec.
        process_group_dict(Dict): A dict which contains the process groups used to apply this CommSpec.
        gather_dim(int, Optional): The gather_dim of the tensor will be gathered.
        shard_dim(int, Optional): The shard_dim of the tensor will be sharded.
        logical_process_axis(Union(int, List[int]), Optional): The mesh_dim to implement the communication action.
    """

    def __init__(
        self,
        comm_pattern: CollectiveCommPattern,
        process_group_dict: Dict,
        gather_dim: int = None,
        shard_dim: int = None,
        logical_process_axis: int = None,
    ):
        self.comm_pattern = comm_pattern
        self.gather_dim = gather_dim
        self.shard_dim = shard_dim
        self.logical_process_axis = logical_process_axis
        self.process_group_dict = process_group_dict

    def __repr__(self):
        res_list = ["CommSpec:("]
        if self.comm_pattern == CollectiveCommPattern.GATHER_FWD_SPLIT_BWD:
            res_list.append(f"comm_pattern:GATHER_FWD_SPLIT_BWD, ")
            res_list.append(f"gather_dim:{self.gather_dim}, ")
            res_list.append(f"shard_dim:{self.gather_dim}, ")
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

        return "".join(res_list)

    def covert_spec_to_action(self, tensor):
        """
        Convert CommSpec into runtime action, implement real collection communication to target tensor.
        The collection communication action is directed by the CommSpec.

        Argument:
            tensor(torch.Tensor): Tensor stored in each device, which could be different in different ranks.
        """
        if self.comm_pattern in pattern_to_func_dict:
            tensor = pattern_to_func_dict[self.comm_pattern](tensor, self)
        else:
            tensor = tensor
        return tensor


def _all_gather(tensor: torch.Tensor, comm_spec: CommSpec):
    """
    Implement all gather operation on device mesh based on information provided by comm_spec.
    """
    process_group = comm_spec.process_group_dict[comm_spec.logical_process_axis]
    world_size = dist.get_world_size(process_group)
    tensor_list = [torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
    # without this contiguous operation, the all gather may get some unexpected results.
    tensor = tensor.contiguous()
    dist.all_gather(tensor_list, tensor, group=process_group)
    output = torch.cat(tuple(tensor_list), comm_spec.gather_dim).contiguous()
    return output


def _split(tensor: torch.Tensor, comm_spec: CommSpec):
    """
    Implement shard operation on device mesh based on information provided by comm_spec.
    """
    process_group = comm_spec.process_group_dict[comm_spec.logical_process_axis]
    dim = comm_spec.shard_dim
    length = tensor.shape[comm_spec.shard_dim] // dist.get_world_size(process_group)
    start = length * dist.get_rank(process_group)
    output = torch.narrow(tensor, dim, start, length).clone().contiguous()
    return output


def _all_to_all(tensor: torch.Tensor, comm_spec: CommSpec):
    """
    Implement all to all operation on device mesh based on information provided by comm_spec.
    """
    process_group = comm_spec.process_group_dict[comm_spec.logical_process_axis]
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


def _all_reduce(tensor: torch.Tensor, comm_spec: CommSpec, async_op: bool = False):
    """
    Implement all reduce operation on device mesh based on information provided by comm_spec.
    """
    process_group = comm_spec.process_group_dict[comm_spec.logical_process_axis]
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    dist.all_reduce(tensor, op=ReduceOp.SUM, group=process_group, async_op=async_op)
    return tensor


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
        comm_spec_for_backward = CommSpec(
            comm_pattern=comm_spec.comm_pattern,
            process_group_dict=comm_spec.process_group_dict,
            gather_dim=comm_spec.shard_dim,
            shard_dim=comm_spec.gather_dim,
            logical_process_axis=comm_spec.logical_process_axis,
        )
        ctx.comm_spec = comm_spec_for_backward
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return _all_to_all(grad_outputs, ctx.comm_spec), None


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


pattern_to_func_dict = {
    CollectiveCommPattern.GATHER_FWD_SPLIT_BWD: gather_forward_split_backward,
    CollectiveCommPattern.ALL2ALL_FWD_ALL2ALL_BWD: all_to_all,
    CollectiveCommPattern.SPLIT_FWD_GATHER_BWD: split_forward_gather_backward,
    CollectiveCommPattern.ALLREDUCE_FWD_IDENTITY_BWD: reduce_input,
    CollectiveCommPattern.IDENTITY_FWD_ALLREDUCE_BWD: reduce_grad,
}
