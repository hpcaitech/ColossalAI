import random

import pytest
import torch
from einops import rearrange

from colossalai.kernel.cuda_native.flash_attention import HAS_MEM_EFF_ATTN

if HAS_MEM_EFF_ATTN:
    from colossalai.kernel.cuda_native.flash_attention import AttnMaskType, ColoAttention


def baseline_attention(Z, N_CTX, H, q, k, v, sm_scale):
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


@pytest.mark.skipif(HAS_MEM_EFF_ATTN == False, reason="xformers is not available")
@pytest.mark.parametrize('B, S, H, D_HEAD', [(6, 8, 4, 16)])
def test_attention_gpt(B, S, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    c_attn = torch.nn.Linear(D, 3 * D, dtype=dtype, device="cuda")
    attn = ColoAttention(D, H, dropout=0.1)

    x = torch.randn((B, S, D), dtype=dtype, device="cuda")

    qkv = c_attn(x)
    q, k, v = rearrange(qkv, 'b s (n h d) -> n b s h d', n=3, h=H)
    y = attn(q, k, v, attn_mask_type=AttnMaskType.causal)

    assert list(y.shape) == [B, S, D]

    dy = torch.rand_like(y)
    y.backward(dy)


@pytest.mark.skipif(HAS_MEM_EFF_ATTN == False, reason="xformers is not available")
@pytest.mark.parametrize('B, S, H, D_HEAD', [(6, 8, 4, 16)])
def test_attention_bert(B, S, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    c_attn = torch.nn.Linear(D, 3 * D, dtype=dtype, device="cuda")
    attn = ColoAttention(D, H, dropout=0.1)

    x = torch.randn((B, S, D), dtype=dtype, device="cuda")
    # attention mask of shape [B, S] with zero padding to max length S
    mask = [torch.ones(S - i, dtype=dtype, device="cuda") for i in range(B)]
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)

    qkv = c_attn(x)
    q, k, v = rearrange(qkv, 'b s (n h d) -> b s n h d', n=3, h=H).unbind(dim=2)
    y = attn(q, k, v, attn_mask=mask, attn_mask_type=AttnMaskType.padding)

    assert list(y.shape) == [B, S, D]

    dy = torch.rand_like(y)
    y.backward(dy)


@pytest.mark.skipif(HAS_MEM_EFF_ATTN == False, reason="xformers is not available")
@pytest.mark.parametrize('B, S, H, D_HEAD', [(6, 8, 4, 16)])
def test_attention_no_mask(B, S, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    c_attn = torch.nn.Linear(D, 3 * D, dtype=dtype, device="cuda")
    attn = ColoAttention(D, H, dropout=0.1)

    x = torch.randn((B, S, D), dtype=dtype, device="cuda")
    qkv = c_attn(x)
    q, k, v = rearrange(qkv, 'b s (n h d) -> b s n h d', n=3, h=H).unbind(dim=2)
    y = attn(q, k, v)

    assert list(y.shape) == [B, S, D]

    dy = torch.rand_like(y)
    y.backward(dy)


@pytest.mark.skipif(HAS_MEM_EFF_ATTN == False, reason="xformers is not available")
@pytest.mark.parametrize('B, S, T, H, D_HEAD', [(6, 24, 8, 4, 16)])
def test_cross_attention(B, S, T, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    q_attn = torch.nn.Linear(D, D, dtype=dtype, device="cuda")
    kv_attn = torch.nn.Linear(D, 2 * D, dtype=dtype, device="cuda")

    attn = ColoAttention(D, H, dropout=0.1)

    src = torch.randn((B, S, D), dtype=dtype, device="cuda")
    tgt = torch.randn((B, T, D), dtype=dtype, device="cuda")

    q = q_attn(tgt)
    kv = kv_attn(src)
    q = rearrange(q, 'b s (h d) -> b s h d', h=H)
    k, v = rearrange(kv, 'b s (n h d) -> b s n h d', n=2, h=H).unbind(dim=2)
    y = attn(q, k, v, attn_mask_type=AttnMaskType.causal)

    assert list(y.shape) == [B, T, D]

    dy = torch.rand_like(y)
    y.backward(dy)
