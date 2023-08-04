from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from colossalai.utils.cuda import get_current_device


class Unpad(torch.autograd.Function):
    """
    Adapted from
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, indices: torch.Tensor):
        ctx.save_for_backward(indices)
        # [b, s, ...]
        assert tensor.ndim >= 3
        ctx.bsz = tensor.shape[0]
        out = rearrange(tensor, 'b s ... -> (b s) ...')
        ctx.shape = out.shape
        # [ntokens, ...]
        return out[indices]

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # [ntokens, ...]
        grad = torch.zeros(ctx.shape, dtype=grad_output.dtype, device=grad_output.device)
        grad[indices] = grad_output
        grad = rearrange(grad, '(b s) ... -> b s ...', b=ctx.bsz)
        # [b, s, ...]
        return grad, None


class Repad(torch.autograd.Function):
    """
    Adapted from
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int):
        ctx.save_for_backward(indices)
        # [ntokens, ...]
        tensor = tensor
        out = torch.zeros((batch_size * seq_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        # [b*s, ...]
        out[indices] = tensor
        return out

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # [b*s, ...]
        grad = grad_output[indices]
        # [ntokens, ...]
        return grad, None, None, None


@dataclass
class SeqLenInfo:
    seqlens: Iterable[int] = None
    indices: torch.Tensor = None
    max_seqlen: int = None
    cu_seqlens: torch.Tensor = None

    @staticmethod
    def materialize(attn_mask: torch.Tensor = None, size: Tuple[int] = None, device=get_current_device()):
        if attn_mask is not None:
            indices = torch.nonzero(attn_mask.flatten(), as_tuple=False).flatten().to(device)
            seqlens = attn_mask.sum(dim=-1, dtype=torch.int32).flatten()
        else:
            batch_size, tgt_len = size[0], size[1]
            indices = torch.arange(batch_size * tgt_len, dtype=torch.long, device=device)
            seqlens = torch.LongTensor([tgt_len] * batch_size, device=device)
        max_seqlen = max(seqlens)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)).to(device)
        return SeqLenInfo(seqlens.tolist(), indices, max_seqlen, cu_seqlens)
