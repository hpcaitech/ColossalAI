import torch

from colossalai.kernel.triton import copy_kv_to_blocked_cache, decoding_fused_rotary_embedding, rotary_embedding
from tests.test_infer.test_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache_v2, mock_alloc_single_token

try:
    import triton  # noqa

except ImportError:
    print("please install triton from https://github.com/openai/triton")


BATCH = 16
configs = [
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[2**i for i in range(4, 11)],
        line_arg="provider",
        line_vals=["no_fused_rotary_emb_func", "fused_triton_rotary_emb_func"],
        line_names=["no_fused_rotary_emb_func", "fused_triton_rotary_emb_func"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"rotary_emb-batch-{BATCH}",
        args={"num_kv_heads": 16},
    )
]


@triton.testing.perf_report(configs)
def benchmark_rotary_emb(
    provider: str,
    num_tokens: int,
    num_kv_heads: int,
):
    BATCH_SIZE = 4
    SEQ_LEN = num_tokens // BATCH_SIZE
    max_num_blocks_per_seq = 8
    block_size = 64
    warmup = 10
    rep = 100

    head_dim = 4096
    dtype = torch.float16

    q_shape = (num_tokens, num_kv_heads, head_dim)
    q = -2.3 + 0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")
    k_shape = (num_tokens, num_kv_heads, head_dim)
    k = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")
    v = -2.3 + 0.5 * torch.randn(k_shape, dtype=dtype, device="cuda")

    cos_shape = (num_tokens, head_dim // 2)

    cos = -1.2 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    sin = -2.0 + 0.5 * torch.randn(cos_shape, dtype=dtype, device="cuda")
    cache_shape = (BATCH_SIZE * max_num_blocks_per_seq, num_kv_heads, block_size, head_dim)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device="cuda")
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device="cuda")

    past_kv_seq_lengths = torch.tensor([SEQ_LEN - 1 for _ in range(BATCH_SIZE)], dtype=torch.int32, device="cuda")
    block_tables = mock_alloc_block_table_and_kvcache_v2(
        k, v, k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_num_blocks_per_seq, block_size
    )
    new_k = torch.randn((BATCH_SIZE, num_kv_heads, head_dim), dtype=dtype, device="cuda")
    new_q = torch.randn_like(new_k)
    new_v = torch.randn_like(new_k)

    mock_alloc_single_token(block_tables, past_kv_seq_lengths, block_size)
    kv_seq_lengths = past_kv_seq_lengths + 1
    block_tables = block_tables.to(device="cuda")

    if provider == "no_fused_rotary_emb_func":
        fn = lambda: [
            rotary_embedding(new_q, new_k, cos, sin),
            copy_kv_to_blocked_cache(
                new_k, new_v, k_cache, v_cache, kv_lengths=kv_seq_lengths, block_tables=block_tables
            ),
        ]
    elif provider == "fused_triton_rotary_emb_func":
        fn = lambda: decoding_fused_rotary_embedding(
            new_q, new_k, new_k, cos, sin, k_cache, k_cache, block_tables, kv_seq_lengths
        )
    else:
        raise ValueError("Undefined provider")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    benchmark_rotary_emb.run(save_path=".", print_data=True)
