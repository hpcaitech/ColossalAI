import pytest
import torch
from packaging import version

from colossalai.inference.quant.smoothquant.models.smoothquant_layer import LLamaSmoothquantAttention
from colossalai.kernel.triton import int8_rotary_embedding_fwd

try:
    from colossalai.inference.quant.smoothquant.models.smoothquant_layer import LLamaSmoothquantAttention
    from colossalai.kernel.triton import int8_rotary_embedding_fwd

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

import math

import torch
from torch.nn import functional as F


def torch_context_attention(xq, xk, xv, bs, seqlen, num_head, head_dim):
    """
    adapted from https://github.com/ModelTC/lightllm/blob/main/lightllm/models/bloom/triton_kernel/context_flashattention_nopad.py#L253
    """
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.0] = -100000000.0
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    sm_scale = 1 / math.sqrt(head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) * sm_scale
    scores = F.softmax(scores.float() + mask, dim=-1).to(dtype=torch.float)

    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
def test_llama_context_attention():
    head_num = 8
    seq_len = 32
    head_dim = 64
    dtype = torch.float
    hidden_size = head_num * head_dim

    smooth_attn = LLamaSmoothquantAttention(head_num * head_dim, head_num)

    smooth_attn.q_proj.weight = torch.ones(hidden_size, hidden_size).to(torch.int8)
    smooth_attn.k_proj.weight = torch.ones(hidden_size, hidden_size).to(torch.int8)
    smooth_attn.v_proj.weight = torch.ones(hidden_size, hidden_size).to(torch.int8)
    smooth_attn.out_proj.weight = torch.ones(hidden_size, hidden_size).to(torch.int8)

    smooth_attn = smooth_attn.to("cuda")

    input = torch.randint(-127, 127, (1, seq_len, head_num * head_dim), dtype=torch.int8, device="cuda")

    q = smooth_attn.q_proj(input)
    k = smooth_attn.k_proj(input)
    v = smooth_attn.v_proj(input)

    cos_shape = (seq_len, head_dim // 2)
    cos = torch.ones(cos_shape, dtype=dtype, device="cuda")
    sin = torch.zeros(cos_shape, dtype=dtype, device="cuda")

    in_scale = torch.tensor([1.0], device="cuda")
    out_scale = torch.tensor([1.0], device="cuda")

    int8_rotary_embedding_fwd(q.view(-1, head_num, head_dim), cos, sin, in_scale, out_scale)
    int8_rotary_embedding_fwd(k.view(-1, head_num, head_dim), cos, sin, in_scale, out_scale)

    q = q.to(torch.float)
    k = k.to(torch.float)
    v = v.to(torch.float)
    torch_out = torch_context_attention(q.clone(), k.clone(), v.clone(), 1, seq_len, head_num, head_dim)
    torch_out = (torch_out).to(torch.int8).view(-1, seq_len, head_num * head_dim)
    torch_out = smooth_attn.out_proj(torch_out)
    smooth_out, _, _ = smooth_attn(input, (cos, sin))
    smooth_out = smooth_out.to(torch.float)
    torch_out = torch_out.to(torch.float)

    assert torch.allclose(
        smooth_out.cpu(), torch_out.cpu(), rtol=1e-2, atol=1e-2
    ), "outputs from triton and torch are not matched"


if __name__ == "__main__":
    test_llama_context_attention()
