from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup

from colossalai.quantization.fp8 import all_to_all_single_fp8

MOE_KERNEL = None


def load_moe():
    global MOE_KERNEL
    from colossalai.kernel.kernel_loader import MoeLoader

    MOE_KERNEL = MoeLoader().load()


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: Optional[ProcessGroup] = None,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.unsqueeze(0), None

        buffer_shape = (comm_size,) + inputs.shape
        outputs = torch.empty(buffer_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(outputs, comm_size, dim=0))
        if not overlap:
            dist.all_gather(buffer_list, inputs, group=group)
            return outputs, None
        else:
            handle = dist.all_gather(buffer_list, inputs, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            ReduceScatter.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.squeeze(0), None

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        output_shape = inputs.shape[1:]
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(inputs, comm_size, dim=0))
        if not overlap:
            dist.reduce_scatter(outputs, buffer_list, group=group)
            return outputs, None
        else:
            handle = dist.reduce_scatter(outputs, buffer_list, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        # TODO: support async backward
        return (
            AllGather.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class AllToAll(torch.autograd.Function):
    """Dispatches input tensor [e, c, h] to all experts by all_to_all_single
    operation in torch.distributed.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if dist.get_world_size(group) == 1:
            return inputs, None
        output = torch.empty_like(inputs)
        if not overlap:
            dist.all_to_all_single(output, inputs, group=group)
            return output, None
        else:
            handle = dist.all_to_all_single(output, inputs, group=group, async_op=True)
            return output, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            AllToAll.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class HierarchicalAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, groups: Tuple[ProcessGroup, ProcessGroup], src_rank: int) -> Tensor:
        """
        Returns:
            outputs: Tensor
        """
        # TODO: we can reduce comm volume by removing empty capacity
        if ctx is not None:
            ctx.comm_grps = groups
            ctx.src_rank = src_rank
        intra_node_group, inter_node_group = groups

        local_world_size = dist.get_world_size(intra_node_group)
        num_group = dist.get_world_size(inter_node_group) if inter_node_group is not None else 1
        world_size = local_world_size * num_group
        outputs = torch.empty_like(inputs)

        if dist.get_rank() == src_rank:
            # intra-node gather
            intra_output = [torch.empty_like(inputs) for _ in range(local_world_size)]
            dist.gather(inputs, intra_output, dst=src_rank, group=intra_node_group)

            intra_output = [v.chunk(world_size, dim=0) for v in intra_output]
            intra_output = torch.cat(sum(zip(*intra_output), ()))

            # inter-node all-to-all
            if inter_node_group is not None:
                inter_output = torch.empty_like(intra_output)
                dist.all_to_all_single(inter_output, intra_output, group=inter_node_group)

                # layout transform
                inter_output = inter_output.chunk(num_group, dim=0)
                inter_output = [v.chunk(local_world_size, dim=0) for v in inter_output]
                intra_output = torch.cat(sum(zip(*inter_output), ()))

            # intra-node scatter
            intra_output = list(intra_output.chunk(local_world_size, dim=0))
            dist.scatter(outputs, intra_output, src=src_rank, group=intra_node_group)

        else:
            dist.gather(inputs, dst=src_rank, group=intra_node_group)
            dist.scatter(outputs, src=src_rank, group=intra_node_group)

        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            HierarchicalAllToAll.forward(None, grad_outputs[0], ctx.comm_grps, ctx.src_rank),
            None,
            None,
        )


class MoeDispatch(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, tokens, mask, dest_idx, ec):
        s = tokens.size(0)
        h = tokens.size(1)
        dtype = tokens.dtype

        if MOE_KERNEL is None:
            load_moe()
        if tokens.dtype != torch.float32:
            tokens = tokens.to(torch.float32)
        expert_input = MOE_KERNEL.dispatch_forward(s, ec, h, tokens, mask, dest_idx)
        if expert_input.dtype != dtype:
            expert_input = expert_input.to(dtype)
        ctx.save_for_backward(mask, dest_idx)
        ctx.s = s
        ctx.h = h
        ctx.ec = ec
        ctx.dtype = dtype

        return expert_input

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        mask, dest_idx = ctx.saved_tensors
        if output_grad.dtype != torch.float32:
            output_grad = output_grad.to(torch.float32)
        d_tokens = MOE_KERNEL.dispatch_backward(ctx.s, ctx.ec, ctx.h, output_grad, mask, dest_idx)
        if d_tokens.dtype != ctx.dtype:
            d_tokens = d_tokens.to(ctx.dtype)
        return d_tokens, None, None, None


class MoeCombine(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, expert_tokens, logits, mask, dest_idx, ec):
        assert logits.dtype == torch.float32

        s = logits.size(0)
        e = logits.size(1)
        c = ec // e
        h = expert_tokens.size(-1)
        dtype = expert_tokens.dtype

        if expert_tokens.dtype != torch.float32:
            expert_tokens = expert_tokens.to(torch.float32)
        if MOE_KERNEL is None:
            load_moe()
        output = MOE_KERNEL.combine_forward(s, e, c, h, expert_tokens, logits, mask, dest_idx)
        if output.dtype != dtype:
            output = output.to(dtype)

        ctx.save_for_backward(expert_tokens, logits, mask, dest_idx)
        ctx.s = s
        ctx.e = e
        ctx.c = c
        ctx.h = h
        ctx.dtype = dtype

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, tokens_grad):
        expert_tokens, logits, mask, dest_idx = ctx.saved_tensors
        if tokens_grad.dtype != torch.float32:
            tokens_grad = tokens_grad.to(torch.float32)

        d_expert, d_logits = MOE_KERNEL.combine_backward(
            ctx.s, ctx.e, ctx.c, ctx.h, tokens_grad, expert_tokens, logits, mask, dest_idx
        )
        if d_expert.dtype != ctx.dtype:
            d_expert = d_expert.to(ctx.dtype)

        return d_expert, d_logits, None, None, None


def moe_cumsum(inputs: Tensor, use_kernel: bool = False):
    dim0 = inputs.size(0)
    flag = (dim0 <= 1024) or (dim0 <= 2048 and dim0 % 2 == 0) or (dim0 % 4 == 0)
    if flag and use_kernel:
        if MOE_KERNEL is None:
            load_moe()
        return MOE_KERNEL.cumsum_sub_one(inputs)
    else:
        return torch.cumsum(inputs, dim=0) - 1


class EPGradScalerIn(torch.autograd.Function):
    """
    Scale the gradient back by the number of experts
    because the batch size increases in the moe stage
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, ep_size: int) -> Tensor:
        ctx.ep_size = ep_size
        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        if ctx.ep_size != 1:
            grad.mul_(ctx.ep_size)
        return grad, None


class EPGradScalerOut(torch.autograd.Function):
    """
    Scale the gradient by the number of experts
    because the batch size increases in the moe stage
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, ep_size: int) -> Tensor:
        ctx.ep_size = ep_size
        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        if ctx.ep_size != 1:
            grad.div_(ctx.ep_size)
        return grad, None


class DPGradScalerIn(torch.autograd.Function):
    """
    Scale the gradient back by the number of experts
    because the batch size increases in the moe stage
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, moe_dp_size: int, activated_experts: int) -> Tensor:
        assert activated_experts != 0, f"shouldn't be called when no expert is activated"
        ctx.moe_dp_size = moe_dp_size
        ctx.activated_experts = activated_experts
        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None, None]:
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        if ctx.moe_dp_size != ctx.activated_experts:
            grad.mul_(ctx.activated_experts / ctx.moe_dp_size)
        return grad, None, None


class DPGradScalerOut(torch.autograd.Function):
    """
    Scale the gradient by the number of experts
    because the batch size increases in the moe stage
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, moe_dp_size: int, activated_experts: int) -> Tensor:
        assert activated_experts != 0, f"shouldn't be called when no expert is activated"
        ctx.moe_dp_size = moe_dp_size
        ctx.activated_experts = activated_experts
        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None, None]:
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        if ctx.moe_dp_size != ctx.activated_experts:
            grad.mul_(ctx.moe_dp_size / ctx.activated_experts)
        return grad, None, None


def _all_to_all(
    inputs: torch.Tensor,
    input_split_sizes: Optional[List[int]] = None,
    output_split_sizes: Optional[List[int]] = None,
    group=None,
    async_op: bool = False,
    fp8_communication: bool = False,
):
    """
    Returns:
        outputs: Tensor
        handle: Optional[Work], if overlap is True
    """
    outputs_shape = list(inputs.shape)
    if output_split_sizes is not None:
        outputs_shape[0] = sum(output_split_sizes)
    outputs = torch.empty(outputs_shape, dtype=inputs.dtype, device=inputs.device)
    inputs = inputs.contiguous()
    outputs = outputs.contiguous()
    if fp8_communication:
        handle = all_to_all_single_fp8(
            outputs, inputs, output_split_sizes, input_split_sizes, group=group, async_op=False
        )
    else:
        handle = dist.all_to_all_single(
            outputs, inputs, output_split_sizes, input_split_sizes, group=group, async_op=async_op
        )
    return outputs, handle


class AllToAllUneven(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        input_split_sizes=None,
        output_split_sizes=None,
        group=None,
        overlap: bool = False,
        fp8_communication: bool = False,
    ):
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        return _all_to_all(
            inputs, input_split_sizes, output_split_sizes, group, overlap, fp8_communication=fp8_communication
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs):
        return (
            _all_to_all(grad_outputs[0], ctx.output_split_sizes, ctx.input_split_sizes, ctx.group, False)[0],
            None,
            None,
            None,
            None,
            None,
        )


def all_to_all_uneven(
    inputs: torch.Tensor,
    input_split_sizes: Optional[List[int]] = None,
    output_split_sizes: Optional[List[int]] = None,
    group=None,
    overlap: bool = False,
    fp8_communication: bool = False,
):
    return AllToAllUneven.apply(inputs, input_split_sizes, output_split_sizes, group, overlap, fp8_communication)
