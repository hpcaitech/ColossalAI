import math

import pytest
import torch
from packaging import version
from torch import nn
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl

    from colossalai.kernel.triton import bloom_context_attn_fwd
    from tests.test_infer_ops.triton.kernel_utils import torch_context_attention
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')


@pytest.mark.skipif(not TRITON_CUDA_SUPPORT or not HAS_TRITON,
                    reason="triton requires cuda version to be higher than 11.4")
def test_bloom_context_attention():
    bs = 4
    head_num = 8
    seq_len = 1024
    head_dim = 64

    query = torch.randn((bs * seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")
    k = torch.randn((bs * seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")
    v = torch.randn((bs * seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")

    max_input_len = seq_len
    b_start = torch.zeros((bs,), device="cuda", dtype=torch.int32)
    b_len = torch.zeros((bs,), device="cuda", dtype=torch.int32)

    for i in range(bs):
        b_start[i] = i * seq_len
        b_len[i] = seq_len

    o = torch.randn((bs * seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")
    alibi = torch.zeros((head_num,), dtype=torch.float32, device="cuda")
    bloom_context_attn_fwd(query.clone(), k.clone(), v.clone(), o, b_start, b_len, max_input_len, alibi)

    torch_out = torch_context_attention(query.clone(), k.clone(), v.clone(), bs, seq_len, head_num, head_dim)

    assert torch.allclose(torch_out.cpu(), o.cpu(), rtol=1e-3,
                          atol=1e-2), "outputs from triton and torch are not matched"


if __name__ == "__main__":
    test_bloom_context_attention()
