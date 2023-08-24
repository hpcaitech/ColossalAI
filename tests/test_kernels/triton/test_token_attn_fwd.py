import time

import torch

from colossalai.kernel.triton.token_attention_kernel import token_attention_fwd


def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim):
    xq = xq.view(bs, 1, num_head, head_dim)
    xk = xk.view(bs, seqlen, num_head, head_dim)
    xv = xv.view(bs, seqlen, num_head, head_dim)

    logics = torch.sum(xq * xk, dim=3, keepdim=False) * 1 / (head_dim**0.5)
    prob = torch.softmax(logics, dim=1)
    prob = prob.view(bs, seqlen, num_head, 1)

    return torch.sum(prob * xv, dim=1, keepdim=False)


def test():

    Z, head_num, seq_len, head_dim = 22, 112 // 8, 2048, 128
    dtype = torch.float16
    q = torch.empty((Z, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * seq_len, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z, head_num, head_dim), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    alibi = torch.zeros((head_num,), dtype=torch.float32, device="cuda")

    max_kv_cache_len = seq_len
    kv_cache_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    kv_cache_loc = torch.zeros((Z, seq_len), dtype=torch.int32, device="cuda")
    kv_cache_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")

    kv_cache_seq_len[:] = seq_len
    kv_cache_start_loc[0] = 0
    kv_cache_start_loc[1] = seq_len
    kv_cache_start_loc[2] = 2 * seq_len
    kv_cache_start_loc[3] = 3 * seq_len

    for i in range(Z):
        kv_cache_loc[i, :] = torch.arange(i * seq_len, (i + 1) * seq_len, dtype=torch.int32, device="cuda")

    token_attention_fwd(q, k, v, o, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_kv_cache_len, alibi=alibi)
    torch.cuda.synchronize()
    start = time.time()
    token_attention_fwd(q, k, v, o, kv_cache_loc, kv_cache_start_loc, kv_cache_seq_len, max_kv_cache_len, alibi=alibi)
    torch.cuda.synchronize()
    print("cost time:", (time.time() - start) * 1000)

    torch_att(q, k, v, Z, seq_len, head_num, head_dim)
    torch.cuda.synchronize()
    start = time.time()
    torch_out = torch_att(q, k, v, Z, seq_len, head_num, head_dim)
    torch.cuda.synchronize()
    print("cost time:", (time.time() - start) * 1000)

    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)


if __name__ == "__main__":
    test()
