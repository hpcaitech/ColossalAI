import pytest
import torch
from packaging import version

try:
    pass

    from colossalai.kernel.triton.token_attention_kernel import token_attn_fwd_2

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def torch_attn(V, P, bs, seqlen, num_head, head_dim):
    V = V.view(bs, seqlen, num_head, head_dim).transpose(1, 2)
    P = P.reshape(num_head, bs, 1, seqlen).transpose(0, 1)
    attn_out = torch.matmul(P, V)

    return attn_out


@pytest.mark.skipif(
    not TRITON_CUDA_SUPPORT or not HAS_TRITON, reason="triton requires cuda version to be higher than 11.4"
)
def test_token_attn_2():
    pass

    batch_size, seq_len, head_num, head_dim = 17, 1025, 12, 128
    dtype = torch.float16

    V = torch.empty((batch_size * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    Prob = (
        torch.empty((head_num, batch_size * seq_len), dtype=dtype, device="cuda")
        .normal_(mean=0.4, std=0.2)
        .reshape(head_num, batch_size, seq_len)
        .softmax(-1)
        .reshape(head_num, batch_size * seq_len)
    )
    attn_out = torch.empty((batch_size, head_num, head_dim), dtype=dtype, device="cuda")

    kv_cache_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    kv_cache_seq_len = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    kv_cache_loc = torch.zeros((batch_size, seq_len), dtype=torch.int32, device="cuda")
    for i in range(batch_size):
        kv_cache_start_loc[i] = i * seq_len
        kv_cache_seq_len[i] = seq_len
        kv_cache_loc[i] = i * seq_len + torch.arange(0, seq_len, dtype=torch.int32, device="cuda")

    token_attn_fwd_2(Prob, V, attn_out, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, seq_len)

    torch_out = torch_attn(V, Prob, batch_size, seq_len, head_num, head_dim).squeeze()
    o = attn_out
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_token_attn_2()
