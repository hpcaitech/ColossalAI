import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from typing import Any, Tuple

U_CUDA_MODE = False
try:
    import colossal_moe_cuda

    U_CUDA_MODE = True
except ImportError:
    print("If you want to activate cuda mode for MoE, please install with cuda_ext!")


class AllToAll(torch.autograd.Function):
    """Dispatches input tensor [e, c, h] to all experts by all_to_all_single
    operation in torch.distributed.
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, parallel_mode: ParallelMode) -> Tensor:
        if ctx is not None:
            ctx.parallel_mode = parallel_mode
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if gpc.get_world_size(parallel_mode) == 1:
            return inputs
        output = torch.empty_like(inputs)
        dist.all_to_all_single(output, inputs, group=gpc.get_group(parallel_mode))
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        return AllToAll.forward(None, *grad_outputs, ctx.parallel_mode), None


class MoeDispatch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, mask, dest_idx, ec):
        s = tokens.size(0)
        h = tokens.size(1)

        expert_input = colossal_moe_cuda.dispatch_forward(s, ec, h, tokens, mask, dest_idx)

        ctx.save_for_backward(mask, dest_idx)
        ctx.s = s
        ctx.h = h
        ctx.ec = ec

        return expert_input

    @staticmethod
    def backward(ctx, output_grad):
        mask, dest_idx = ctx.saved_tensors
        d_tokens = colossal_moe_cuda.dispatch_backward(ctx.s, ctx.ec, ctx.h, output_grad, mask, dest_idx)
        return d_tokens, None, None, None


class MoeCombine(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expert_tokens, logits, mask, dest_idx, ec):
        assert logits.dtype == torch.float32

        s = logits.size(0)
        e = logits.size(1)
        c = ec // e
        h = expert_tokens.size(-1)

        fp16_flag = (expert_tokens.dtype == torch.float16)
        cb_input = expert_tokens.to(torch.float32) if fp16_flag else expert_tokens
        ctokens = colossal_moe_cuda.combine_forward(s, e, c, h, cb_input, logits, mask, dest_idx)
        output = ctokens.to(torch.float16) if fp16_flag else ctokens

        ctx.save_for_backward(expert_tokens, logits, mask, dest_idx)
        ctx.s = s
        ctx.e = e
        ctx.c = c
        ctx.h = h
        ctx.fp16_flag = fp16_flag

        return output

    @staticmethod
    def backward(ctx, tokens_grad):
        expert_tokens, logits, mask, dest_idx = ctx.saved_tensors

        cb_grad = tokens_grad.to(torch.float32) if tokens_grad.dtype is torch.float16 \
            else tokens_grad
        cb_input = expert_tokens.to(torch.float32) if ctx.fp16_flag else expert_tokens
        d_expert, d_logits = colossal_moe_cuda.combine_backward(ctx.s, ctx.e, ctx.c, ctx.h, cb_grad, cb_input, logits,
                                                                mask, dest_idx)
        d_expert = d_expert.to(torch.float16) if ctx.fp16_flag else d_expert

        return d_expert, d_logits, None, None, None


def moe_cumsum(inputs: Tensor):
    dim0 = inputs.size(0)
    flag = (dim0 <= 1024) or (dim0 <= 2048 and dim0 % 2 == 0) or (dim0 % 4 == 0)
    if flag and U_CUDA_MODE:
        return colossal_moe_cuda.cumsum_sub_one(inputs)
    else:
        return torch.cumsum(inputs, dim=0) - 1
