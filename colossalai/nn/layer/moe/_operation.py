import torch
import torch.distributed as dist
from torch import Tensor

from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from typing import Any, Tuple


class AllToAll(torch.autograd.Function):
    """Dispatches input tensor [e, c, h] to all experts by all_to_all_single
    operation in torch.distributed.
    """
    @staticmethod
    def forward(ctx: Any,
                inputs: Tensor,
                parallel_mode: ParallelMode) -> Tensor:
        ctx.parallel_mode = parallel_mode
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        output = torch.empty_like(inputs)
        dist.all_to_all_single(output, inputs,
                               group=gpc.get_group(parallel_mode))
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        return AllToAll.apply(*grad_outputs, ctx.parallel_mode), None
