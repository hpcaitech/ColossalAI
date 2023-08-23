import pytest
import math
from packaging import version

import torch
from torch import nn
from torch.nn import functional as F


from tests.test_kernels.triton.utils import benchmark
from colossalai.kernel.triton.bloom_context_attention import bloom_context_flash_attention_fwd as bloom_context_attn_fwd

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse('11.4')

def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim):
    '''
     adepted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/bloom/triton_kernel/context_flashattention_nopad.py#L253
    '''
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    sm_scale = 1/math.sqrt(head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) * sm_scale
    scores = F.softmax(scores.float() + mask, dim=-1).to(dtype=torch.float16)
    
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output

@pytest.mark.skipif(not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4")
def test_bloom_context_attention():
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
    alibi = torch.zeros((head_num,), dtype=torch.float32, device="cuda")
    bloom_context_attn_fwd(query.clone(), k.clone(), v.clone(), o, alibi, b_start, b_len,max_input_len)
    
    torch_out = torch_att(query.clone(), k.clone(), v.clone(), bs, seq_len, head_num, head_dim)
    
    assert torch.allclose(torch_out.cpu(), o.cpu(), rtol=1e-3, atol=1e-2), "outputs from triton and torch are not matched"
    
    latency_1 = benchmark(bloom_context_attn_fwd, query, k, v, o, alibi, b_start, b_len, max_input_len)
    latency_2 = benchmark(torch_att, query, k, v, bs, seq_len, head_num, head_dim)
    
    print("the triton op latency is {} ms".format(str(latency_1)))
    print("the torch op latency is {} ms".format(str(latency_2)))
    

if __name__ == "__main__":
    test_bloom_context_attention()