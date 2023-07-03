import torch
import torch.nn as nn
from torch.testing import assert_close
from transformers.models.opt.modeling_opt import OPTAttention

from colossalai.shardformer.layer import FlashAttentionForOPT


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
    attention_mask = torch.ones((4, 1, 4096, 4096), dtype=torch.float32, device="cuda")

    opt_attention = OPTAttention(embed_dim=D_HEAD * N_HEADS, num_heads=N_HEADS, dropout=0, is_decoder=True,
                                 bias=True).to("cuda")
    opt_flash_attention = FlashAttentionForOPT().from_native_module(opt_attention, None).to("cuda")

    opt_attention_output = opt_attention(hidden_states, key_value_states, attention_mask=attention_mask)
    flash_attention_output = opt_flash_attention(hidden_states, key_value_states, attention_mask=attention_mask)

    assert_close(flash_attention_output, opt_attention_output, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    test_flash_attention_for_opt()
