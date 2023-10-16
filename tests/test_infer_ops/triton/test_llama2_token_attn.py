import pytest
import torch
from packaging import version

try:
    pass

    from colossalai.kernel.triton.token_attention_kernel import Llama2TokenAttentionForwards

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, 1, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)

    logics = torch.sum(xq * xk, dim=3, keepdim=False) * 1 / (head_dim**0.5)
    prob = torch.softmax(logics, dim=1)
    prob = prob.view(bs, seqlen, num_head, 1)

    return torch.sum(prob * xv, dim=1, keepdim=False)


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
def test():
    Z, head_num, seq_len, head_dim = 2, 32, 2048, 128
    dtype = torch.float16

    # attn out: 2,4096
    q = torch.empty((Z, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z, head_num, head_dim), dtype=dtype, device="cuda")
    max_kv_cache_len = seq_len
    kv_cache_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    kv_cache_loc = torch.zeros((Z, seq_len), dtype=torch.int32, device="cuda")
    kv_cache_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    other_kv_index = 2048

    kv_cache_seq_len[:] = seq_len
    kv_cache_start_loc[0] = 0
    kv_cache_start_loc[1] = seq_len

    for i in range(Z):
        kv_cache_loc[i, :] = torch.arange(i * seq_len, (i + 1) * seq_len, dtype=torch.int32, device="cuda")

    Llama2TokenAttentionForwards.token_attn(
        q, k, v, o, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_kv_cache_len, other_kv_index
    )
    torch_out = torch_att(q, k, v, Z, seq_len, head_num, head_dim)
    assert torch.allclose(torch_out, o, atol=1e-3, rtol=0)


if __name__ == "__main__":
    test()
