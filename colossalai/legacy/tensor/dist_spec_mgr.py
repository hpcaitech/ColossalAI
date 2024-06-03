from contextlib import contextmanager

import torch
import torch.distributed as dist
from numpy import prod

from colossalai.legacy.tensor.distspec import DistPlacementPattern, _DistSpec
from colossalai.legacy.tensor.process_group import ProcessGroup


# TODO(jiaruifang) circle import, move the divide to colossalai.commons.
# colossalai.legacy.tensor shall not import any submodule from colossal.nn
def divide(numerator, denominator):
    """Only allow exact division.

    Args:
        numerator (int): Numerator of the division.
        denominator (int): Denominator of the division.

    Returns:
        int: the result of exact division.
    """
    assert denominator != 0, "denominator can not be zero"
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator


class TransformDistSpec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, old_dist_spec, dist_spec, pg, forward_trans_func, backward_trans_func):
        ctx.old_dist_spec = old_dist_spec
        ctx.dist_spec = dist_spec
        ctx.backward_trans_func = backward_trans_func
        ctx.pg = pg
        return forward_trans_func(tensor, old_dist_spec, dist_spec, pg)

    @staticmethod
    def backward(ctx, grad_outputs):
        return (
            ctx.backward_trans_func(grad_outputs, ctx.dist_spec, ctx.old_dist_spec, ctx.pg),
            None,
            None,
            None,
            None,
            None,
        )


class DistSpecManager:
    _use_autograd_function: bool = True

    @staticmethod
    def _sanity_check(old_dist_spec: _DistSpec, dist_spec: _DistSpec) -> None:
        pass

    @staticmethod
    def _shard_as(
        tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup
    ) -> torch.Tensor:
        """_shard_as: shard the tensor w.r.t a distributed specification.
        Assuming the tensor passed in is a global (replicated) tensor.
        Args:
            tensor (torch.Tensor): a global (replicated) tensor before shard
            dist_spec (_DistSpec): the distributed spec. to be sharded as.
            pg (ProcessGroup): the process group of the corresponding colotensor
        Returns:
            torch.Tensor: a torch tensor after sharded.
        """
        assert (
            old_dist_spec.placement.value == "r"
        ), f"The old_dist_spec of DistSpecManager._shard_as must be REPLICATE!"
        DistSpecManager._sanity_check(old_dist_spec, dist_spec)

        chunk = tensor
        idx = pg.tp_local_rank()
        num_parts = prod(dist_spec.num_partitions)
        for i, dim in enumerate(dist_spec.dims):
            num_parts //= dist_spec.num_partitions[i]

            chunk_size = divide(tensor.size(dim), dist_spec.num_partitions[i])
            chunk = chunk.narrow(dim, idx // num_parts * chunk_size, chunk_size)
            idx %= num_parts
        return chunk.clone().detach().contiguous()

    @staticmethod
    def _gather(tensor: torch.Tensor, old_dist_spec: _DistSpec, pg: ProcessGroup) -> torch.Tensor:
        """_gather gather sharded tensors to a replicated one.
        Args:
            tensor (torch.Tensor): a shared torch tensor
            old_dist_spec (_DistSpec): the distributed spec. of the tensor.

        Returns:
            torch.Tensor: a replicated tensor.
        """
        assert old_dist_spec.placement.value == "s", f"The old_dist_spec of DistSpecManager._gather must be SHARD!"
        is_cpu_tensor = False
        if tensor.device.type == "cpu":
            # pytorch lower than 1.11 dose not support gather a cpu tensor.
            # Therefore, we transfer tensor to GPU before gather.
            saved_dev = tensor.device
            tensor.data = tensor.data.cuda()
            is_cpu_tensor = True

        buffer = [torch.empty_like(tensor) for _ in range(pg.tp_world_size())]
        assert tensor.device.type == "cuda"
        dist.all_gather(buffer, tensor, group=pg.tp_process_group())
        for i in range(len(old_dist_spec.dims) - 1, -1, -1):
            new_buffer = []
            dim = old_dist_spec.dims[i]
            num_parts = old_dist_spec.num_partitions[i]
            for start in range(0, len(buffer), num_parts):
                new_buffer.append(torch.cat(buffer[start : start + num_parts], dim))
            buffer = new_buffer
        assert len(buffer) == 1

        if is_cpu_tensor:
            buffer[0].data = buffer[0].data.to(saved_dev)
        return buffer[0]

    @staticmethod
    def _all_to_all(
        tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup
    ) -> torch.Tensor:
        world_size = pg.tp_world_size()
        if world_size == 1:
            return tensor

        assert tensor.device.type == "cuda", (
            "Currently, only CUDA Tensor with NCCL backend is supported for the requested AlltoAll "
            f"collective function, however, we got {tensor.device.type} device"
        )

        gather_dim = old_dist_spec.dims[0]
        scatter_dim = dist_spec.dims[0]
        shapes = list(tensor.shape)
        scattered_dim_size = shapes[scatter_dim] // world_size
        gathered_dim_size = shapes[gather_dim] * world_size
        shapes[scatter_dim] = scattered_dim_size

        scatter_list = [t.contiguous() for t in torch.tensor_split(tensor, world_size, scatter_dim)]
        gather_list = [torch.empty(*shapes, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
        dist.all_to_all(gather_list, scatter_list, group=pg.tp_process_group())

        output_ = torch.cat(gather_list, dim=gather_dim).contiguous()
        assert output_.shape[scatter_dim] == scattered_dim_size and output_.shape[gather_dim] == gathered_dim_size
        return output_

    @staticmethod
    def _r2r(tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup) -> torch.Tensor:
        DistSpecManager._sanity_check(old_dist_spec, dist_spec)
        return tensor

    @staticmethod
    def _r2s(tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup) -> torch.Tensor:
        DistSpecManager._sanity_check(old_dist_spec, dist_spec)
        return DistSpecManager._shard_as(tensor, old_dist_spec, dist_spec, pg)

    @staticmethod
    def _s2r(tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup) -> torch.Tensor:
        DistSpecManager._sanity_check(old_dist_spec, dist_spec)
        return DistSpecManager._gather(tensor, old_dist_spec, pg)

    @staticmethod
    def _s2s(tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup) -> torch.Tensor:
        DistSpecManager._sanity_check(old_dist_spec, dist_spec)
        if old_dist_spec == dist_spec:
            return tensor
        if len(old_dist_spec.dims) == 1 and len(dist_spec.dims) == 1:
            # use all-to-all to save memory
            return DistSpecManager._all_to_all(tensor, old_dist_spec, dist_spec, pg)
        tensor = DistSpecManager._gather(tensor, old_dist_spec, pg)
        return DistSpecManager._shard_as(tensor, old_dist_spec, dist_spec, pg)

    @staticmethod
    def handle_trans_spec(
        tensor: torch.Tensor, old_dist_spec: _DistSpec, dist_spec: _DistSpec, pg: ProcessGroup
    ) -> torch.Tensor:
        assert isinstance(old_dist_spec, _DistSpec), f"{type(old_dist_spec)} should be _DistSpec"
        assert isinstance(dist_spec, _DistSpec), f"{type(dist_spec)} should be _DistSpec"

        trans_func_key = (old_dist_spec.placement, dist_spec.placement)
        trans_funcs = {
            (DistPlacementPattern.REPLICATE, DistPlacementPattern.REPLICATE): DistSpecManager._r2r,
            (DistPlacementPattern.REPLICATE, DistPlacementPattern.SHARD): DistSpecManager._r2s,
            (DistPlacementPattern.SHARD, DistPlacementPattern.REPLICATE): DistSpecManager._s2r,
            (DistPlacementPattern.SHARD, DistPlacementPattern.SHARD): DistSpecManager._s2s,
        }

        forward_trans_handle = trans_funcs[trans_func_key]
        if not DistSpecManager._use_autograd_function:
            return forward_trans_handle(tensor, old_dist_spec, dist_spec, pg)

        backward_trans_handle = trans_funcs[(dist_spec.placement, old_dist_spec.placement)]

        return TransformDistSpec.apply(
            tensor, old_dist_spec, dist_spec, pg, forward_trans_handle, backward_trans_handle
        )

    @staticmethod
    @contextmanager
    def no_grad():
        try:
            DistSpecManager._use_autograd_function = False
            yield
        finally:
            DistSpecManager._use_autograd_function = True
