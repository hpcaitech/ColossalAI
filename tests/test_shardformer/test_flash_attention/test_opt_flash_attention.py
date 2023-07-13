from copy import deepcopy
from typing import Optional

import torch
from torch.testing import assert_close
from transformers.models.opt.modeling_opt import OPTAttention

from colossalai.shardformer.modeling.opt import get_opt_forward


def _make_causal_mask(input_ids_shape: torch.Size,
                      dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def test_flash_attention_for_opt():
    BATCH, N_HEADS, N_CTX, D_HEAD = 4, 8, 16, 16

    # generate input
    hidden_states = torch.randn((BATCH, N_CTX, N_HEADS * D_HEAD),
                                dtype=torch.float32,
                                device="cuda",
                                requires_grad=True)
    key_value_states = torch.randn((BATCH, N_CTX, N_HEADS * D_HEAD),
                                   dtype=torch.float32,
                                   device="cuda",
                                   requires_grad=True)
    attention_mask = torch.ones((BATCH, N_CTX), dtype=torch.float32, device="cuda")
    attention_mask = _expand_mask(attention_mask, dtype=torch.float32, tgt_len=N_CTX)
    casual_attention_mask = _make_causal_mask((BATCH, N_CTX), dtype=torch.float32, device="cuda")
    combined_attention_mask = attention_mask + casual_attention_mask

    opt_attention = OPTAttention(embed_dim=D_HEAD * N_HEADS, num_heads=N_HEADS, dropout=0, is_decoder=False,
                                 bias=True).to("cuda")

    opt_flash_attention = deepcopy(opt_attention)
    setattr(opt_flash_attention, 'forward',
            get_opt_forward().__get__(opt_flash_attention, opt_flash_attention.__class__))
    opt_attention_output = opt_attention(hidden_states, key_value_states, attention_mask=combined_attention_mask)
    flash_attention_output = opt_flash_attention(hidden_states,
                                                 key_value_states,
                                                 attention_mask=combined_attention_mask)
    assert opt_attention_output[0].size() == flash_attention_output[0].size()
