import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from ..scaled_softmax import AttnMaskType
from .flash_attn_2 import HAS_FLASH_ATTN
from .mem_eff_attn import HAS_MEM_EFF_ATTN
from .utils import Repad, SeqLenInfo, Unpad

if HAS_FLASH_ATTN:
    from .flash_attn_2 import flash_attention
if HAS_MEM_EFF_ATTN:
    from .mem_eff_attn import mem_eff_attention


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

        if not HAS_MEM_EFF_ATTN and not HAS_FLASH_ATTN:
            raise Exception("flash attention can not support!")

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

        attn = None
        if HAS_FLASH_ATTN and query.dtype in [torch.float16, torch.bfloat16] and bias == None:
            attn = flash_attention
        else:
            attn = mem_eff_attention

        padded = attn_mask_type is not None and attn_mask_type.value % 2 == 1
        causal = attn_mask_type is not None and attn_mask_type.value > 1

        batch_size, tgt_len, src_len = query.shape[0], query.shape[1], key.shape[1]
        # unpad
        seq_len_info_q = None
        seq_len_info_kv = None
        if padded:
            # bert style, unpad process
            assert attn_mask is not None, \
                f"attention mask {attn_mask} is not valid for attention mask type {attn_mask_type}."
            assert attn_mask.dim() == 2, \
                "attention mask is supposed to have shape (batch_size, seq_len), " + \
                f"but got {attn_mask.dim()} dimensions."

            # bert style
            if tgt_len == src_len:
                seq_len_info_q = SeqLenInfo.materialize(attn_mask=attn_mask, device=query.device)
                if batch_size > 1:
                    query, key, value = self.unpad(torch.stack([query, key, value], dim=2),
                                                   seq_len_info_q.indices).unbind(dim=1)
                else:
                    query, key, value = torch.stack([query, key, value], dim=2).squeeze(0).unbind(dim=1)
                seq_len_info_kv = seq_len_info_q
            else:
                seq_len_info_q = SeqLenInfo.materialize(size=(batch_size, tgt_len), device=query.device)
                seq_len_info_kv = SeqLenInfo.materialize(attn_mask=attn_mask, device=query.device)
                if batch_size > 1:
                    query = rearrange(query, "b s ... -> c (b s) ...", c=1)
                    key, value = self.unpad(torch.stack([query, key, value], dim=2),
                                            seq_len_info_kv.indices).unbind(dim=1)
                else:
                    query, key, value = torch.stack([query, key, value], dim=2).squeeze(0).unbind(dim=1)

        out = attn(query,
                   key,
                   value,
                   seq_len_info_q,
                   seq_len_info_kv,
                   dropout_p=self.dropout,
                   scale=self.scale,
                   causal=causal,
                   padded=padded)

        # repad
        if padded:
            if batch_size > 1:
                out = self.repad(out, seq_len_info_q.indices, batch_size, tgt_len)
            out = rearrange(out, '(b s) h d -> b s h d', b=batch_size)

        out = rearrange(out, 'b s h d -> b s (h d)')
        return out
