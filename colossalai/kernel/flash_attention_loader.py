import math
from typing import Optional

import torch
from einops import rearrange

from .base_kernel_loader import BaseKernelLoader
from .extensions.flash_attention import CudaFlashAttnExtension, CudaMemoryEfficentAttnExtension
from .extensions.utils import AttnMaskType, Repad, SeqLenInfo, Unpad


class FlashAttentionLoader(BaseKernelLoader):
    """
    FlashAttention Loader
    """

    def __init__(self):
        super().__init__(
            extension_map=dict(
                cuda_flash_attn=CudaFlashAttnExtension,
                cuda_memory_efficent_attn=CudaMemoryEfficentAttnExtension,
            ),
            supported_device=["cuda", "npu"],
        )

    def fetch_kernel(self, backend: str = None):
        if backend is not None:
            return self._extension_map[backend].fetch()

        kernel = None
        for _, kernel_extension in self._extension_map.items():
            ext = kernel_extension()
            if ext.is_available():
                kernel = ext.fetch()
                break
        if kernel is None:
            raise Exception("not supported")
        return kernel


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

        self.attn = FlashAttentionLoader().fetch_kernel()

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
        # if flash attention is not applicable, switch to memory effcient attention
        if self.attn.__name__ == "flash_attention" and (
            query.dtype not in [torch.float16, torch.bfloat16] or bias != None
        ):
            self.attn = FlashAttentionLoader().fetch_kernel(backend="cuda_memory_efficent_attn")

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
            seq_len_info_q,
            seq_len_info_kv,
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

        out = rearrange(out, "b s h d -> b s (h d)")
        return out
