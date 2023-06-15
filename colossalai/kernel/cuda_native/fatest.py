"""
A general attention module using the flash attention kernels from xformers:
https://github.com/facebookresearch/xformers/tree/main/xformers/ops/fmha
"""

import math
import os
import subprocess

import torch

try:
    #from xformers.ops.fmha import memory_efficient_attention
    from colossalai.kernel.cuda_native.flash_attn.ops.fmha import memory_efficient_attention
    HAS_MEM_EFF_ATTN = True
    print("HAS_MEM_EFF_ATTN = True****************************")
    
except ImportError:
    HAS_MEM_EFF_ATTN = False
    print('please install xformers from https://github.com/facebookresearch/xformers')

if HAS_MEM_EFF_ATTN:

    from typing import Optional
    print("HAS_MEM_EFF_ATTN = True into func")
    from einops import rearrange
    #from xformers.ops.fmha import MemoryEfficientAttentionCutlassOp
    from colossalai.kernel.cuda_native.flash_attn.ops.fmha import MemoryEfficientAttentionCutlassOp

    #from xformers.ops.fmha.attn_bias import BlockDiagonalMask, LowerTriangularMask, LowerTriangularMaskWithTensorBias
    from colossalai.kernel.cuda_native.flash_attn.ops.fmha.attn_bias import BlockDiagonalMask, LowerTriangularMask, LowerTriangularMaskWithTensorBias

    #from .scaled_softmax import AttnMaskType
    import enum

    class AttnMaskType(enum.Enum):
        padding = 1
        causal = 2

    allow_alibi = True
    print(" allow_alibi = True****************************")
    for op in MemoryEfficientAttentionCutlassOp:
        allow_alibi = allow_alibi & (LowerTriangularMaskWithTensorBias in op.SUPPORTED_ATTN_BIAS_TYPES)

    class Unpad(torch.autograd.Function):
        """
        Adapted from
        https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
        """

        @staticmethod
        def forward(ctx, tensor: torch.Tensor, indices: torch.Tensor):
            ctx.save_for_backward(indices)
            # [b, s, ...]
            assert tensor.ndim >= 3
            ctx.bsz = tensor.shape[0]
            out = rearrange(tensor, 'b s ... -> (b s) ...')
            ctx.shape = out.shape
            # [1, ntokens, ...]
            return out[indices].unsqueeze(0)

        @staticmethod
        def backward(ctx, grad_output):
            indices, = ctx.saved_tensors
            # [b*s, ...]
            grad = torch.zeros(ctx.shape, dtype=grad_output.dtype, device=grad_output.device)
            grad[indices] = grad_output.squeeze(0)
            grad = rearrange(grad, '(b s) ... -> b s ...', b=ctx.bsz)
            # [b, s, ...]
            return grad, None

    class Repad(torch.autograd.Function):
        """
        Adapted from
        https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
        """

        @staticmethod
        def forward(ctx, tensor: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int):
            ctx.save_for_backward(indices)
            # [ntokens, ...]
            tensor = tensor.squeeze(0)
            out = torch.zeros((batch_size * seq_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            # [b*s, ...]
            out[indices] = tensor
            # [b, s, ...]
            out = rearrange(out, '(b s) ... -> b s ...', b=batch_size)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            indices, = ctx.saved_tensors
            # [b*s, ...]
            grad_output = rearrange(grad_output, 'b s ... -> (b s) ...')
            grad = grad_output[indices]
            # [1, ntokens, ...]
            return grad.unsqueeze(0), None, None, None

    class ColoAttention(torch.nn.Module):

        def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
            print("*************def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0)***************")
            super().__init__()
            assert embed_dim % num_heads == 0, \
                f"the embed dim ({embed_dim}) is not divisible by the number of attention heads ({num_heads})."
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
            print("flash attn forward****************************")
            batch_size, tgt_len, src_len = query.shape[0], query.shape[1], key.shape[1]
            attn_bias = None
            if attn_mask_type == AttnMaskType.padding:    # bert style
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
                attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
            elif attn_mask_type == AttnMaskType.causal:    # gpt style
                attn_bias = LowerTriangularMask()

            if bias is not None:    # alibi / relative position emebedding
                assert allow_alibi, "flash attention with bias is not supported in this system."
                assert attn_mask_type == AttnMaskType.causal, \
                    "attention with bias is only supported for causal attention so far."
                attn_bias = attn_bias.add_bias(bias)

            out = memory_efficient_attention(query, key, value, attn_bias=attn_bias, p=self.dropout, scale=self.scale)

            if attn_mask_type == AttnMaskType.padding and batch_size > 1:
                out = self.repad(out, q_indices, batch_size, tgt_len)

            out = rearrange(out, 'b s h d -> b s (h d)')
            return out

######################################

import random

import pytest
import torch
from einops import rearrange

from colossalai.kernel.cuda_native.flash_attention import HAS_MEM_EFF_ATTN
from colossalai.testing import clear_cache_before_run, parameterize

if HAS_MEM_EFF_ATTN:
    print("if HAS_MEM_EFF_ATTN****************************")
    #from colossalai.kernel.cuda_native.flash_attention import AttnMaskType, ColoAttention


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
@clear_cache_before_run()
#@parameterize('B, S, H, D_HEAD', [(6, 8, 4, 16)])
def test_attention_gpt(B, S, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    c_attn = torch.nn.Linear(D, 3 * D, dtype=dtype, device="cuda")
    attn = ColoAttention(D, H, dropout=0.1)
    print("********test_attention_gpt******")

    x = torch.randn((B, S, D), dtype=dtype, device="cuda")

    qkv = c_attn(x)
    q, k, v = rearrange(qkv, 'b s (n h d) -> n b s h d', n=3, h=H)
    y = attn(q, k, v, attn_mask_type=AttnMaskType.causal)

    assert list(y.shape) == [B, S, D]

    dy = torch.rand_like(y)
    y.backward(dy)


@pytest.mark.skipif(HAS_MEM_EFF_ATTN == False, reason="xformers is not available")
@clear_cache_before_run()
#@parameterize('B, S, H, D_HEAD', [(6, 8, 4, 16)])
def test_attention_bert(B, S, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    c_attn = torch.nn.Linear(D, 3 * D, dtype=dtype, device="cuda")
    attn = ColoAttention(D, H, dropout=0.1)
    print("********test_attention_bert******")

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
@clear_cache_before_run()
#@parameterize('B, S, H, D_HEAD', [(6, 8, 4, 16)])
def test_attention_no_mask(B, S, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    c_attn = torch.nn.Linear(D, 3 * D, dtype=dtype, device="cuda")
    attn = ColoAttention(D, H, dropout=0.1)
    print("********test_attention_no_mask******")
    x = torch.randn((B, S, D), dtype=dtype, device="cuda")
    qkv = c_attn(x)
    q, k, v = rearrange(qkv, 'b s (n h d) -> b s n h d', n=3, h=H).unbind(dim=2)
    y = attn(q, k, v)

    assert list(y.shape) == [B, S, D]

    dy = torch.rand_like(y)
    y.backward(dy)


@pytest.mark.skipif(HAS_MEM_EFF_ATTN == False, reason="xformers is not available")
@clear_cache_before_run()
#@parameterize('B, S, T, H, D_HEAD', [(6, 24, 8, 4, 16)])
def test_cross_attention(B, S, T, H, D_HEAD, dtype=torch.float16):
    D = H * D_HEAD

    q_attn = torch.nn.Linear(D, D, dtype=dtype, device="cuda")
    kv_attn = torch.nn.Linear(D, 2 * D, dtype=dtype, device="cuda")

    attn = ColoAttention(D, H, dropout=0.1)
    print("********test_cross_attention******")
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

if __name__ == '__main__':
    
    test_attention_gpt(6,8,4,16)
    test_attention_bert(6,8,4,16)
    test_attention_no_mask(6,8,4,16)
    test_cross_attention(6,24,8,4,16)