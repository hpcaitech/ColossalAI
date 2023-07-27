import math
from typing import Optional

import torch
from einops import rearrange

from ..scaled_softmax import AttnMaskType
from .padding import Repad, Unpad


class ColoAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, scale=None):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"the embed dim ({embed_dim}) is not divisible by the number of attention heads ({num_heads})."
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1 / math.sqrt(embed_dim // num_heads)
        self.dropout = dropout

    @staticmethod
    def get_seq_info_from_mask(attn_mask: torch.Tensor):
        indices = torch.nonzero(attn_mask.flatten(), as_tuple=False).flatten()
        seqlens = attn_mask.sum(dim=-1, dtype=torch.int32).flatten().tolist()
        return indices, seqlens

    @staticmethod
    def unpad(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return Unpad.apply(tensor, indices)

    @staticmethod
    def repad(tensor: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return Repad.apply(tensor, indices, batch_size, seq_len)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                attn_mask_type: Optional[AttnMaskType] = None,
                bias: Optional[torch.Tensor] = None):

        batch_size, tgt_len, src_len = query.shape[0], query.shape[1], key.shape[1]

        import pathlib
        pathlib.Path(
            "/home/lcjmy/code/personal/ColossalAI/colossalai/kernel/cuda_native/ops/flash_attention_2.txt").write_text(
                str(query))

        q_seqlen = None
        kv_seqlen = None
        q_indices = None
        if attn_mask_type and attn_mask_type.value % 2 == 1:    # bert style
            assert attn_mask is not None, \
                f"attention mask {attn_mask} is not valid for attention mask type {attn_mask_type}."
            assert attn_mask.dim() == 2, \
                "attention mask is supposed to have shape (batch_size, seq_len), " + \
                f"but got {attn_mask.dim()} dimensions."
            if tgt_len == src_len:
                q_indices, q_seqlen = self.get_seq_info_from_mask(attn_mask)
                kv_seqlen = None
                if batch_size > 1:
                    query, key, value = self.unpad(torch.stack([query, key, value], dim=2), q_indices).unbind(dim=2)
            else:
                q_indices = torch.arange(batch_size * tgt_len, dtype=torch.int32, device=query.device)
                q_seqlen = torch.LongTensor([tgt_len] * batch_size, device=query.device)
                kv_indices, kv_seqlen = self.get_seq_info_from_mask(attn_mask)
                if batch_size > 1:
                    query = rearrange(query, "b s ... -> c (b s) ...", c=1)
                    key, value = self.unpad(torch.stack([query, key, value], dim=2), kv_indices).unbind(dim=2)

        out = get_attention_output(query,
                                   key,
                                   value,
                                   q_seqlen,
                                   kv_seqlen,
                                   attn_mask_type=attn_mask_type,
                                   bias=bias,
                                   dropout=self.dropout,
                                   scale=self.scale)

        if attn_mask_type and attn_mask_type.value % 2 == 1 and batch_size > 1:
            out = self.repad(out, q_indices, batch_size, tgt_len)

        out = rearrange(out, 'b s h d -> b s (h d)')
        return out


def get_attention_output(query, key, value, q_seqlen, kv_seqlen, attn_mask_type, bias, dropout, scale):
    from .version_available import HAS_FLASH_ATTN, HAS_MEM_EFF_ATTN, HAS_TRITON

    # TODO deal with dispath better
    HAS_MEM_EFF_ATTN = False
    if HAS_MEM_EFF_ATTN:
        from .mem_eff_attn import mem_eff_attention
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            attn_mask_type=attn_mask_type,
            bias=bias,
            dropout=dropout,
            scale=scale,
        )
        attention_output = mem_eff_attention(**kwargs)
        return attention_output
    if HAS_FLASH_ATTN:
        from .flash_attn_2 import flash_attention_q_k_v
        kwargs = dict(q=query,
                      k=key,
                      v=value,
                      sm_scale=scale,
                      cu_seqlens_q=q_seqlen,
                      cu_seqlens_kv=kv_seqlen,
                      dropout_p=dropout,
                      causal=(attn_mask_type == AttnMaskType.causal))
        attention_output = flash_attention_q_k_v(**kwargs)
        return attention_output
    if HAS_TRITON:
        from .triton_flash_attn import triton_flash_attention
        kwargs = dict(
            q=query,
            k=key,
            v=value,
            sm_scale=scale,
        )
        attention_output = triton_flash_attention(**kwargs)
        return attention_output
