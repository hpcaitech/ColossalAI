import warnings
from typing import Optional

import torch


def is_ampere_or_better_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(device)
        if properties.major >= 8:    # Ampere GPUs or newer
            return True
    return False


# "Check Ampere GPUs or newer"
HAS_FLASH_ATTN = False
if is_ampere_or_better_gpu():
    HAS_FLASH_ATTN = True
else:
    warnings.warn('FlashAttention only supports Ampere GPUs or newer.')
    HAS_FLASH_ATTN = False
try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    warnings.warn('please install flash_attn from https://github.com/HazyResearch/flash-attention')
    HAS_FLASH_ATTN = False

if HAS_FLASH_ATTN:
    from einops import rearrange

    from .utils import SeqLenInfo

    def flash_attention(q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        seq_len_info_q: SeqLenInfo,
                        seq_len_info_kv: SeqLenInfo,
                        bias: Optional[torch.Tensor] = None,
                        dropout_p: float = 0.,
                        scale: float = None,
                        causal: bool = False,
                        padded: bool = False):
        """
        Arguments:
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
        if padded:
            if seq_len_info_kv == None:
                seq_len_info_kv = seq_len_info_q

            attn_out = flash_attn_varlen_func(q, k, v, seq_len_info_q.cu_seqlens, seq_len_info_kv.cu_seqlens,
                                              seq_len_info_q.max_seqlen, seq_len_info_kv.max_seqlen, dropout_p, scale,
                                              causal)
        else:
            attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=scale, causal=causal)
        return attn_out
