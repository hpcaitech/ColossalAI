from copy import deepcopy

import torch
import torch.nn as nn
from torch.testing import assert_close
from transformers.models.opt.modeling_opt import OPTAttention

from colossalai.shardformer.layer import opt_flash_attention_forward


def test_flash_attention_for_opt():
    BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64

    # generate input
    hidden_states = torch.randn((BATCH, N_CTX, N_HEADS * D_HEAD),
                                dtype=torch.float32,
                                device="cuda",
                                requires_grad=True)
    key_value_states = torch.randn((BATCH, N_CTX, N_HEADS * D_HEAD),
                                   dtype=torch.float32,
                                   device="cuda",
                                   requires_grad=True)
    attention_mask = torch.ones((BATCH, 1, N_CTX, N_CTX), dtype=torch.float32, device="cuda")

    opt_attention = OPTAttention(embed_dim=D_HEAD * N_HEADS, num_heads=N_HEADS, dropout=0, is_decoder=True,
                                 bias=True).to("cuda")

    opt_flash_attention = deepcopy(opt_attention)
    setattr(opt_flash_attention, 'forward',
            opt_flash_attention_forward.__get__(opt_flash_attention, opt_flash_attention.__class__))
    opt_attention_output = opt_attention(hidden_states, key_value_states, attention_mask=attention_mask)
    flash_attention_output = opt_flash_attention(hidden_states, key_value_states, attention_mask=attention_mask)

    assert_close(flash_attention_output[0], opt_attention_output[0], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    test_flash_attention_for_opt()
