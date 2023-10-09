import pytest
import torch
from packaging import version

try:
    from colossalai.kernel.triton import int8_rotary_embedding_fwd

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

try:
    from colossalai.inference.quant.smoothquant.models import LLamaSmoothquantAttention

    HAS_TORCH_INT = True
except ImportError:
    HAS_TORCH_INT = False
    print("Please install torch_int from https://github.com/Guangxuan-Xiao/torch-int")


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
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)

    return output


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON or not HAS_TORCH_INT,
    reason="triton requires cuda version to be higher than 11.4 or not install torch_int",
)
def test_llama_context_attention():
    head_num = 2
    seq_len = 32
    head_dim = 64
    dtype = torch.float
    hidden_size = head_num * head_dim

    smooth_attn = LLamaSmoothquantAttention(head_num * head_dim, head_num)

    smooth_attn.q_proj.weight = torch.ones(hidden_size, hidden_size, device="cuda").to(torch.int8)
    smooth_attn.k_proj.weight = torch.ones(hidden_size, hidden_size, device="cuda").to(torch.int8)
    smooth_attn.v_proj.weight = torch.ones(hidden_size, hidden_size, device="cuda").to(torch.int8)
    smooth_attn.out_proj.weight = torch.ones(hidden_size, hidden_size, device="cuda").to(torch.int8)
    smooth_attn.out_proj.weight[:, 1:hidden_size] = torch.zeros(hidden_size - 1, device="cuda").to(torch.int8)

    qkv_weight_scale = 1.0

    ones = torch.ones(hidden_size, hidden_size, dtype=torch.float, device="cuda")

    smooth_attn = smooth_attn.to("cuda")

    input = torch.randint(-20, 20, (1, seq_len, head_num * head_dim), dtype=torch.int8, device="cuda")
    input_scale = 1 / 20.0

    output = torch.matmul(input.to(torch.float) * input_scale, ones)
    qkv_max_out = torch.max(torch.abs(output)) / 127
    smooth_attn.q_proj.a = torch.tensor(input_scale * qkv_weight_scale / qkv_max_out)
    smooth_attn.k_proj.a = torch.tensor(input_scale * qkv_weight_scale / qkv_max_out)
    smooth_attn.v_proj.a = torch.tensor(input_scale * qkv_weight_scale / qkv_max_out)

    q = smooth_attn.q_proj(input)
    k = smooth_attn.k_proj(input)
    v = smooth_attn.v_proj(input)

    cos_shape = (seq_len, head_dim // 2)
    cos = torch.ones(cos_shape, dtype=dtype, device="cuda")
    sin = torch.zeros(cos_shape, dtype=dtype, device="cuda")
    in_scale = torch.tensor([qkv_max_out], device="cuda")
    out_scale = torch.tensor([qkv_max_out], device="cuda")
    int8_rotary_embedding_fwd(q.view(-1, head_num, head_dim), cos, sin, in_scale.item(), out_scale.item())
    int8_rotary_embedding_fwd(k.view(-1, head_num, head_dim), cos, sin, in_scale.item(), out_scale.item())

    q = q.to(torch.float) * out_scale
    k = k.to(torch.float) * out_scale
    v = v.to(torch.float) * out_scale
    torch_out = torch_context_attention(q.clone(), k.clone(), v.clone(), 1, seq_len, head_num, head_dim)
    attn_out_max = torch.max(torch.abs(torch_out)) / 127

    output = torch.matmul(torch_out.view(-1, seq_len, head_num * head_dim), ones)
    smooth_attn.q_output_scale = torch.tensor(qkv_max_out)
    smooth_attn.k_output_scale = torch.tensor(qkv_max_out)

    smooth_attn.v_output_scale = torch.tensor(qkv_max_out)
    smooth_attn.q_rotary_output_scale = torch.tensor(qkv_max_out)
    smooth_attn.k_rotary_output_scale = torch.tensor(qkv_max_out)

    smooth_attn.attn_output_scale = torch.tensor(attn_out_max)
    smooth_attn.out_proj.a = torch.tensor([attn_out_max])

    torch_out = (
        (torch_out / smooth_attn.attn_output_scale)
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
        .view(-1, seq_len, head_num * head_dim)
    )

    torch_out = smooth_attn.out_proj(torch_out)
    torch_out = torch_out.to(torch.float)

    smooth_attn = smooth_attn.to("cuda")
    smooth_out, _, _ = smooth_attn(input, (cos, sin))
    smooth_out = smooth_out.to(torch.float)

    assert torch.allclose(
        torch_out.cpu(), smooth_out.cpu(), rtol=1e-1, atol=1e-1
    ), "outputs from triton and torch are not matched"


if __name__ == "__main__":
    test_llama_context_attention()
