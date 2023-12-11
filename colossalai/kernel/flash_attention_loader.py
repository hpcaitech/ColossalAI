import math
from collections import OrderedDict
from typing import Optional

import torch
from einops import rearrange

from colossalai.accelerator import get_accelerator

from .base_kernel_loader import BaseKernelLoader
from .extensions.flash_attention import (
    AttnMaskType,
    CudaFlashAttnExtension,
    CudaMemoryEfficentAttnExtension,
    NpuSdpaAttnExtension,
    NpuTriangleAttnExtension,
    Repad,
    SeqLenInfo,
    Unpad,
)
from .extensions.utils import print_rank_0


class FlashAttentionLoader(BaseKernelLoader):
    """
    FlashAttention Loader

    options: cuda flashh attention, cuda memory effcient attention, npu sdpa attention, npu triangle attention

    Args:
        q: (batch, q_seqlen, nheads, headdim)
        k: (batch, kv_seqlen, nheads, headdim)
        v: (batch, kv_seqlen, nheads, headdim)
        batch_size: int.
        seq_len: int.
        dropout_p: float. Dropout probability.
        sm_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    Return:
        attn_out: (batch, q_seqlen, nheads, headdim).
    """

    def __init__(self):
        super().__init__(
            # extension name must start with the accelerator name. E.g. npu_xxx, cuda_xxx
            extension_map=OrderedDict(
                cuda_flash_attn=CudaFlashAttnExtension,
                cuda_memory_efficent_attn=CudaMemoryEfficentAttnExtension,
                npu_sdpa_attn=NpuSdpaAttnExtension,
                npu_triangle_attn=NpuTriangleAttnExtension,
            ),
            supported_device=["cuda", "npu"],
        )

    def fetch_kernel(self, backend: str = None):
        if backend is not None:
            if not self._extension_map[backend]().is_available():
                raise Exception(f"{backend} is not available for flash attention.")
            return self._extension_map[backend]().fetch()

        kernel = None
        accelerator_name = get_accelerator().name
        assert accelerator_name in self._supported_device, f"{accelerator_name} is not supported for flash attention."
        for extension_name, extension in self._extension_map.items():
            if extension_name.startswith(accelerator_name):
                if extension().is_available():
                    kernel = extension().fetch()
                    break
        if kernel is None:
            raise Exception("No extension for flash attention is supported")
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
            print_rank_0("flash attention is not applicable, switch to memory effcient attention")
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
