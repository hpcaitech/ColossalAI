import math

import torch

from colossalai.kernel.triton.token_attention_kernel import token_attn_fwd_1


def torch_attn(xq, xk, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, 1, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    keys = xk
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    scores = (torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)).squeeze().transpose(0, 1).reshape(
        num_head, -1)
    # print("s  ", scores.shape)
    return scores


def torch_attn_1(xq, xk, seqlen, num_head, head_dim):
    xq = xq.view(1, num_head, head_dim)
    xk = xk.view(seqlen, num_head, head_dim)
    logics = torch.sum(xq * xk, dim=-1, keepdim=False)

    logics = logics.transpose(0, 1) / math.sqrt(head_dim)
    return logics


def test_attn_1():
    import time

    batch_size, seq_len, head_num, head_dim = 17, 1025, 12, 128

    dtype = torch.float16

    q = torch.empty((batch_size, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((batch_size * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    attn_out = torch.empty((head_num, batch_size * seq_len), dtype=dtype, device="cuda")

    # print(attn_out)

    b_loc = torch.zeros((batch_size, seq_len), dtype=torch.int32, device="cuda")
    kv_cache_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
    kv_cache_seq_len = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        kv_cache_start_loc[i] = i * seq_len
        kv_cache_seq_len[i] = seq_len
        b_loc[i] = i * seq_len + torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
        # print(b_loc[i])

    # Warm up
    for _ in range(10):
        token_attn_fwd_1(q, k, attn_out, b_loc, kv_cache_start_loc, kv_cache_seq_len, seq_len)
    run_iter = 1000
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(run_iter):
        token_attn_fwd_1(q, k, attn_out, b_loc, kv_cache_start_loc, kv_cache_seq_len, seq_len)
    torch.cuda.synchronize()
    t2 = time.time()
    print("Time cost {}".format((t2 - t1) / run_iter))

    torch_out = torch_attn(q, k, batch_size, seq_len, head_num, head_dim).squeeze()
    o = attn_out.squeeze()
    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


# def test_alibi_attn_1():
#     import torch

#     batch_size, seq_len, head_num, head_dim = 2, 1025, 12, 128

#     dtype = torch.float16

#     q = torch.empty((batch_size, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
#     k = torch.empty((batch_size * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
#     attn_out = torch.empty((head_num, batch_size * seq_len), dtype=dtype, device="cuda")

#     # print(attn_out)

#     b_loc = torch.zeros((batch_size, seq_len), dtype=torch.int32, device="cuda")
#     kv_cache_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
#     kv_cache_seq_len = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")

#     for i in range(batch_size):
#         kv_cache_start_loc[i] = i * seq_len
#         kv_cache_seq_len[i] = seq_len
#         b_loc[i] = i * seq_len + torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
#         # print(b_loc[i])

#     token_attn_fwd_1(q, k, attn_out, b_loc, kv_cache_start_loc, kv_cache_seq_len, seq_len)

#     torch_out = torch_attn(q, k, batch_size, seq_len, head_num, head_dim).squeeze()
#     o = attn_out.squeeze()
#     print("max ", torch.max(torch.abs(torch_out - o)))
#     print("mean ", torch.mean(torch.abs(torch_out - o)))
#     assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)

if __name__ == "__main__":
    test_attn_1()
    test_alibi_attn_1()
