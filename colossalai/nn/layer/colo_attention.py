import enum
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from colossalai.accelerator import get_accelerator
from colossalai.kernel.kernel_loader import FlashAttentionLoader


@dataclass
class SeqLenInfo:
    seqlens: Iterable[int] = None
    indices: torch.Tensor = None
    max_seqlen: int = None
    cu_seqlens: torch.Tensor = None

    @staticmethod
    def materialize(
        attn_mask: torch.Tensor = None, size: Tuple[int] = None, device=get_accelerator().get_current_device()
    ):
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


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
    paddedcausal = 3


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
        out = rearrange(tensor, "b s ... -> (b s) ...")
        ctx.shape = out.shape
        # [ntokens, ...]
        return out[indices]

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # [ntokens, ...]
        grad = torch.zeros(ctx.shape, dtype=grad_output.dtype, device=grad_output.device)
        grad[indices] = grad_output
        grad = rearrange(grad, "(b s) ... -> b s ...", b=ctx.bsz)
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
        (indices,) = ctx.saved_tensors
        # [b*s, ...]
        grad = grad_output[indices]
        # [ntokens, ...]
        return grad, None, None, None


class ColoAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, scale=None):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"the embed dim ({embed_dim}) is not divisible by the number of attention heads ({num_heads})."
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(embed_dim // num_heads)
        self.dropout = dropout

        self.attn = FlashAttentionLoader().load()

    @staticmethod
    def unpad(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return Unpad.apply(tensor, indices)

    @staticmethod
    def repad(tensor: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return Repad.apply(tensor, indices, batch_size, seq_len)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        origin_attn_mask: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[AttnMaskType] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        ColoAttention

        Args:
            q: (batch, q_seqlen, nheads, headdim)
            k: (batch, kv_seqlen, nheads, headdim)
            v: (batch, kv_seqlen, nheads, headdim)
            origin_attn_mask: (nheads, q_seqlen, kv_seqlen)
            bias: will not be used
        Return:
            attn_out: (batch, q_seqlen, nheads, headdim).
        """
        # if flash attention is not applicable, switch to memory effcient attention
        if self.attn.__name__ == "flash_attention" and (
            query.dtype not in [torch.float16, torch.bfloat16] or bias != None
        ):
            warnings.warn(
                f"flash-attn expects fp16 or bf16 but got {query.dtype}, switching to xformers' implementation."
            )
            self.attn = FlashAttentionLoader().load(ext_name="flash_attention_xformers_cuda")

        padded = attn_mask_type is not None and attn_mask_type.value % 2 == 1
        causal = attn_mask_type is not None and attn_mask_type.value > 1

        batch_size, tgt_len, src_len = query.shape[0], query.shape[1], key.shape[1]
        # unpad
        seq_len_info_q = None
        seq_len_info_kv = None
        if padded:
            # bert style, unpad process
            assert (
                attn_mask is not None
            ), f"attention mask {attn_mask} is not valid for attention mask type {attn_mask_type}."
            assert attn_mask.dim() == 2, (
                "attention mask is supposed to have shape (batch_size, seq_len), "
                + f"but got {attn_mask.dim()} dimensions."
            )

            # bert style
            if tgt_len == src_len:
                seq_len_info_q = SeqLenInfo.materialize(attn_mask=attn_mask, device=query.device)
                if batch_size > 1:
                    query, key, value = self.unpad(
                        torch.stack([query, key, value], dim=2), seq_len_info_q.indices
                    ).unbind(dim=1)
                else:
                    query, key, value = torch.stack([query, key, value], dim=2).squeeze(0).unbind(dim=1)
                seq_len_info_kv = seq_len_info_q
            else:
                seq_len_info_q = SeqLenInfo.materialize(size=(batch_size, tgt_len), device=query.device)
                seq_len_info_kv = SeqLenInfo.materialize(attn_mask=attn_mask, device=query.device)
                if batch_size > 1:
                    query = rearrange(query, "b s ... -> c (b s) ...", c=1)
                    key, value = self.unpad(torch.stack([query, key, value], dim=2), seq_len_info_kv.indices).unbind(
                        dim=1
                    )
                else:
                    query, key, value = torch.stack([query, key, value], dim=2).squeeze(0).unbind(dim=1)

        out = self.attn(
            query,
            key,
            value,
            seq_len_info_q=seq_len_info_q,
            seq_len_info_kv=seq_len_info_kv,
            origin_attn_mask=origin_attn_mask,
            dropout_p=self.dropout,
            scale=self.scale,
            causal=causal,
            padded=padded,
        )

        # repad
        if padded:
            if batch_size > 1:
                out = self.repad(out, seq_len_info_q.indices, batch_size, tgt_len)
            out = rearrange(out, "(b s) h d -> b s h d", b=batch_size)

        if len(out.shape) == 4:
            out = rearrange(out, "b s h d -> b s (h d)")
        return out
