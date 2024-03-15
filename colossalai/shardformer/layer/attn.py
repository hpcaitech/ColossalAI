from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from colossalai.kernel.kernel_loader import (
    FlashAttentionLoader,
    FlashAttentionWithCustomMaskLoader,
    FlashAttentionWithPaddingMaskLoader,
)

__all__ = [
    "AttnMaskType",
    "ColoAttention",
]


class AttnMaskType(Enum):
    CUSTOM = 0
    PADDED = 1
    CAUSAL = 2
    PADDED_CAUSAL = 3


def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    """Invert the mask tensor.

    Args:
        mask (torch.Tensor): Mask tensor. Shape should be [B, 1, Sq, Skv]

    Returns:
        torch.Tensor: Inverted mask tensor.
    """
    inverted_mask = 1.0 - mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(mask.dtype).min)


# adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
def get_pad_info(padding_mask: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Get padding information from padding mask.

    Args:
        padding_mask (torch.Tensor): Padding mask tensor. Shape should be [B, S]

    Returns:
        Tuple[int, torch.Tensor, torch.Tensor]: Tuple of (max_seq_len, cu_seqlens, indices)
    """
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return max_seqlen_in_batch, cu_seqlens, indices


class ColoAttention:
    # these two attrs are initialized in the first call of attention() method
    _flash_attn_func: Optional[Callable] = None
    _flash_attn_with_custom_mask_func: Optional[Callable] = None
    _flash_attn_with_padding_mask_func: Optional[Callable] = None

    @staticmethod
    def _init_flash_attn_func():
        if ColoAttention._flash_attn_func is None:
            ColoAttention._flash_attn_func = FlashAttentionLoader().load()
        if ColoAttention._flash_attn_with_custom_mask_func is None:
            ColoAttention._flash_attn_with_custom_mask_func = FlashAttentionWithCustomMaskLoader().load()
        if ColoAttention._flash_attn_with_padding_mask_func is None:
            ColoAttention._flash_attn_with_padding_mask_func = FlashAttentionWithPaddingMaskLoader().load()

    @staticmethod
    def prepare_attn_kwargs(
        shape_4d: Tuple[int],
        dtype: torch.dtype,
        device: torch.device,
        q_padding_mask: Optional[torch.Tensor] = None,
        kv_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return a dictionary of keyword arguments for attention function. It supports 4 mask type.
        1. custom mask: no padding mask and is_causal=False, return {}, users should handle attention mask by themselves.
        2. padded mask: recv padding mask and is_causal=False, return {attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, q_indices, kv_indices}.
        3. causal mask: no padding mask and is_causal=True, return {attention_mask, attention_mask_type}.
        4. padded causal mask: recv padding mask and is_causal=True, return {attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, q_indices, kv_indices}.

        Args:
            shape_4d (Tuple[int]): Should be (B, 1, Sq, Skv)
            dtype (torch.dtype): Dtype of attention mask, generally should be ``hidden_states.dtype``
            device (torch.device): Device of attention mask, generally should be ``hidden_states.device``
            q_padding_mask (Optional[torch.Tensor], optional): Padding mask of query. It should be a long tensor or int tensor.
                The shape should be [B, Sq]. ``1`` means valid token, and ``0`` means padding token. Defaults to None.
            kv_padding_mask (Optional[torch.Tensor], optional): Padding mask of key and value. It should be a long tensor or int tensor.
                The shape should be [B, Skv]. ``1`` means valid token, and ``0`` means padding token.
                If it's None and ``q_padding_mask`` is not None, it will be set to ``q_padding_mask``. Defaults to None.
            is_causal (bool, optional): Whether to use causal attention mask. Defaults to False.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of keyword arguments for attention function.
        """
        if q_padding_mask is None and not is_causal:
            return {}
        assert len(shape_4d) == 4 and shape_4d[1] == 1
        b, _, s_q, s_kv = shape_4d
        outputs = {}
        if q_padding_mask is not None:
            if kv_padding_mask is None:
                kv_padding_mask = q_padding_mask
            assert q_padding_mask.shape == (b, s_q) and kv_padding_mask.shape == (b, s_kv)
            attention_mask = torch.einsum("bi,bj->bij", q_padding_mask, kv_padding_mask).to(dtype=dtype, device=device)
            max_seqlen_q, cu_seqlens_q, q_indices = get_pad_info(q_padding_mask)
            max_seqlen_kv, cu_seqlens_kv, kv_indices = get_pad_info(kv_padding_mask)
            outputs.update(
                {
                    "cu_seqlens_q": cu_seqlens_q,
                    "cu_seqlens_kv": cu_seqlens_kv,
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_kv": max_seqlen_kv,
                    "q_indices": q_indices,
                    "kv_indices": kv_indices,
                }
            )
            if is_causal:
                outputs["attention_mask_type"] = AttnMaskType.PADDED_CAUSAL
                attention_mask = attention_mask * attention_mask.new_ones(s_q, s_kv).tril(diagonal=0)
            else:
                outputs["attention_mask_type"] = AttnMaskType.PADDED
        else:
            assert is_causal
            outputs["attention_mask_type"] = AttnMaskType.CAUSAL
            attention_mask = torch.ones(s_q, s_kv, dtype=dtype, device=device).tril(diagonal=0).expand(b, s_q, s_kv)
        attention_mask = invert_mask(attention_mask).unsqueeze(1)
        outputs["attention_mask"] = attention_mask
        return outputs

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_type: AttnMaskType = AttnMaskType.CUSTOM,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        q_indices: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Flash Attention function. It supports 4 mask type.
        1. custom mask: recv attention_mask
        2. padded mask: recv attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, indices
        3. causal mask: recv attention_mask, attention_mask_type
        4. padded causal mask: recv attention_mask, attention_mask_type, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, indices

        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, N, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, N, Skv, D]
            v (torch.Tensor): Value tensor. Shape should be [B, N, Skv, D]
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor. Shape should be [B, 1, Sq, Skv]. Defaults to None.
            attention_mask_type (AttnMaskType, optional): Attention mask type. Defaults to AttnMaskType.CUSTOM.
            cu_seqlens_q (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
                Shape should be [B+1]. Defaults to None.
            cu_seqlens_kv (Optional[torch.Tensor], optional): The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
                Shape should be [B+1]. Defaults to None.
            max_seqlen_q (Optional[int], optional): Maximum query sequence length in the batch. Defaults to None.
            max_seqlen_kv (Optional[int], optional): Maximum key/value sequence length in the batch. Defaults to None.
            indices (Optional[torch.Tensor], optional): The indices of non-masked tokens from the flattened input sequence.
                Shape should be [NUM_TOKENS]. Defaults to None.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            scale (Optional[float], optional): Scaling factor applied prior to softmax. Defaults to None.

        Returns:
            torch.Tensor: Output tensor. Shape should be [B, N, Sq, D]
        """
        ColoAttention._init_flash_attn_func()
        # known issue: sdpa does not support attention mask which contains whole row of masked tokens, which leads to nan
        # this case is usaul when padding mask is used and self attention is performed
        # thus, we don't use sdpa when padding mask is used
        # sanity check
        if attention_mask is not None:
            assert torch.is_floating_point(attention_mask), "attention_mask should be a floating point tensor."
            if attention_mask_type in (AttnMaskType.CUSTOM, AttnMaskType.CAUSAL):
                assert (
                    cu_seqlens_q is None
                    and cu_seqlens_kv is None
                    and max_seqlen_q is None
                    and max_seqlen_kv is None
                    and q_indices is None
                    and kv_indices is None
                )
                if attention_mask_type == AttnMaskType.CUSTOM:
                    assert not torch.all(attention_mask != 0, dim=-1).any()
            elif attention_mask_type in (AttnMaskType.PADDED, AttnMaskType.PADDED_CAUSAL):
                assert (
                    cu_seqlens_q is not None
                    and cu_seqlens_kv is not None
                    and max_seqlen_q is not None
                    and max_seqlen_kv is not None
                    and q_indices is not None
                    and kv_indices is not None
                )
        else:
            # if attention_mask is None, attention_mask_type should be the default value
            assert attention_mask_type == AttnMaskType.CUSTOM
        # kernel dispatch
        if attention_mask is not None and attention_mask_type == AttnMaskType.CUSTOM:
            attn_func = ColoAttention._flash_attn_with_custom_mask_func
        elif attention_mask_type in (AttnMaskType.PADDED, AttnMaskType.PADDED_CAUSAL):
            attn_func = ColoAttention._flash_attn_with_padding_mask_func
        else:
            attn_func = ColoAttention._flash_attn_func
        is_causal = attention_mask is not None and attention_mask_type in (
            AttnMaskType.CAUSAL,
            AttnMaskType.PADDED_CAUSAL,
        )
        return attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            scale=scale,
            attention_mask=attention_mask,
            is_causal=is_causal,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            q_indices=q_indices,
            kv_indices=kv_indices,
        )
