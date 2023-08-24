from colossalai.kernel.triton.token_attention_kernel import token_attn_softmax_fwd


def test_softmax():

    import torch

    batch_size, seq_len, head_num, head_dim = 4, 1025, 12, 128

    dtype = torch.float16

    Logics = torch.empty((head_num, batch_size * seq_len), dtype=dtype, device="cuda").normal_(mean=0.1, std=10)
    ProbOut = torch.empty((head_num, batch_size * seq_len), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)

    kv_cache_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    kv_cache_seq_len = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        kv_cache_start_loc[i] = i * seq_len
        kv_cache_seq_len[i] = seq_len

    token_attn_softmax_fwd(Logics, kv_cache_start_loc, kv_cache_seq_len, ProbOut, seq_len)

    torch_out = Logics.reshape(head_num * batch_size, -1).softmax(-1).reshape(head_num, batch_size * seq_len)
    o = ProbOut
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test_softmax()
