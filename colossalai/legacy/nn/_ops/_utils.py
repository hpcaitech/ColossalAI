from typing import List, Optional, Union

import torch
import torch.distributed as dist

from colossalai.legacy.global_variables import tensor_parallel_env as env
from colossalai.legacy.nn.layer.utils import divide
from colossalai.legacy.tensor import ColoTensorSpec, ProcessGroup
from colossalai.tensor import ColoTensor

GeneralTensor = Union[ColoTensor, torch.Tensor]
Number = Union[int, float]


def convert_to_colo_tensor(tensor: Optional[GeneralTensor], pg: ProcessGroup) -> Optional[ColoTensor]:
    if tensor is not None and not isinstance(tensor, ColoTensor):
        tensor = ColoTensor.from_torch_tensor(tensor, ColoTensorSpec(pg))
    return tensor


def set_parallel_input(input_parallel: bool):
    env.parallel_input_1d = input_parallel


def get_parallel_input():
    return env.parallel_input_1d


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank)


def _reduce(input_, pg: ProcessGroup):
    # skip if only one rank involved
    if pg.tp_world_size() == 1:
        return input_
    assert input_.device.type == "cuda"
    group = pg.tp_process_group()
    dist.all_reduce(input_, group=group)

    return input_


def _split(input_, pg: ProcessGroup, dim=-1):
    # skip if only one rank involved
    world_size = pg.tp_world_size()
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = pg.tp_local_rank()
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_, pg: ProcessGroup, dim=-1):
    # skip if only one rank involved
    world_size = pg.tp_world_size()
    if world_size == 1:
        return input_

    # all gather
    rank = pg.tp_local_rank()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    assert input_.device.type == "cuda"
    group = pg.tp_process_group()
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _ReduceGrad(torch.autograd.Function):
    """
    Pass the input to the model parallel region.

    Args:
        input_: input matrix.
        process_group: parallel mode.
    """

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.mode = process_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.mode), None


class _ReduceInput(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.

    Args:
        input_: input matrix.
        process_group: parallel mode.
    """

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_, process_group):
        return _reduce(input_, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim):
        ctx.mode = process_group
        ctx.dim = dim
        return _split(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim):
        ctx.mode = process_group
        ctx.dim = dim
        return _gather(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mode, ctx.dim), None, None


def reduce_grad(input_, process_group):
    return _ReduceGrad.apply(input_, process_group)


def reduce_input(input_, process_group):
    return _ReduceInput.apply(input_, process_group)


def split_forward_gather_backward(input_, process_group, dim):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim)


def gather_forward_split_backward(input_, process_group, dim):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim)


def _all_to_all(x: torch.Tensor, pg: ProcessGroup, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    world_size = pg.tp_world_size()
    if world_size == 1:
        return x

    # TODO: enabling mpi backend to support CPU all_to_all
    assert x.device.type == "cuda", f"Currently, the collective function dual_all_to_all only supports nccl backend"

    shapes = list(x.size())
    shapes[scatter_dim] = shapes[scatter_dim] // world_size

    scatter_list = [each.contiguous() for each in torch.tensor_split(x, world_size, scatter_dim)]
    gather_list = [torch.empty(*shapes, dtype=x.dtype, device=x.device) for _ in range(world_size)]
    torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group())

    return torch.cat(gather_list, dim=gather_dim).contiguous()


class _DualAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pg, scatter_dim, gather_dim):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.pg = pg
        return _all_to_all(x, pg, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad):
        return _all_to_all(grad, ctx.pg, ctx.gather_dim, ctx.scatter_dim), None, None, None


def dual_all_to_all(x, pg, scatter_dim: int, gather_dim: int):
    return _DualAllToAll.apply(x, pg, scatter_dim, gather_dim)


# table wise embedding shard


def _all_to_all_for_tablewise(
    x: torch.Tensor, pg: ProcessGroup, scatter_strides: List[int], gather_strides: List[int], forward=True
) -> torch.Tensor:
    world_size = pg.tp_world_size()
    rank = pg.tp_local_rank()
    if world_size == 1:
        return x
    assert x.device.type == "cuda", f"Currently, the collective function dual_all_to_all only supports nccl backend"
    if forward:
        scatter_list = list(x.split(scatter_strides, 0))
        gather_list = [
            torch.empty(scatter_strides[rank], gather_strides[i], dtype=x.dtype, device=x.device)
            for i in range(world_size)
        ]
        torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group())
        return torch.cat(gather_list, 1).contiguous()
    else:
        # split on dim 1, lose contiguity
        scatter_list = [each.contiguous() for each in x.split(scatter_strides, 1)]
        gather_list = [
            torch.empty(gather_strides[i], scatter_strides[rank], dtype=x.dtype, device=x.device)
            for i in range(world_size)
        ]
        torch.distributed.all_to_all(gather_list, scatter_list, group=pg.tp_process_group())
        return torch.cat(gather_list, 0).contiguous()


class _DualAllToAllForTablewise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pg, scatter_strides, gather_strides):
        ctx.pg = pg
        ctx.scatter_strides = scatter_strides
        ctx.gather_strides = gather_strides
        return _all_to_all_for_tablewise(x, pg, scatter_strides, gather_strides, forward=True)

    @staticmethod
    def backward(ctx, grad):
        return (
            _all_to_all_for_tablewise(grad, ctx.pg, ctx.gather_strides, ctx.scatter_strides, forward=False),
            None,
            None,
            None,
        )


def dual_all_to_all_tablewise(x, pg, scatter_strides, gather_strides):
    return _DualAllToAllForTablewise.apply(x, pg, scatter_strides, gather_strides)
