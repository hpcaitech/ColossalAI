import pytest
import math
from packaging import version

import torch
from torch import nn
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
    from tests.test_infer_ops.triton.utils import benchmark, torch_context_attention
    from colossalai.kernel.triton import llama_context_attn_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')


@pytest.mark.skipif(not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4")
def test_llama_context_attention():
    bs = 4
    head_num = 8
    seq_len = 1024
    head_dim = 64
    
    query = torch.randn((bs*seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")
    k = torch.randn((bs*seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")
    v = torch.randn((bs*seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")

    
    max_input_len = seq_len
    b_start = torch.zeros((bs, ), device="cuda", dtype=torch.int32)
    b_len = torch.zeros((bs, ), device="cuda", dtype=torch.int32)
    
    for i in range(bs):
        b_start[i] = i * seq_len
        b_len[i] = seq_len
    
    o = torch.randn((bs*seq_len, head_num, head_dim), dtype=torch.float16, device="cuda")
    llama_context_attn_fwd(query.clone(), k.clone(), v.clone(), o, b_start, b_len, max_input_len)
    
    torch_out = torch_context_attention(query.clone(), k.clone(), v.clone(), bs, seq_len, head_num, head_dim)
    
    assert torch.allclose(torch_out.cpu(), o.cpu(), rtol=1e-3, atol=1e-2), "outputs from triton and torch are not matched"
    
    latency_1 = benchmark(llama_context_attn_fwd, query, k, v, o, b_start, b_len, max_input_len)
    latency_2 = benchmark(torch_context_attention, query, k, v, bs, seq_len, head_num, head_dim)
    
    print("the triton op latency is {} ms".format(str(latency_1)))
    print("the torch op latency is {} ms".format(str(latency_2)))
    

if __name__ == "__main__":
    test_llama_context_attention()