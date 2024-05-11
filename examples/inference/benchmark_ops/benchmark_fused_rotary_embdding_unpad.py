import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import copy_kv_to_blocked_cache, decoding_fused_rotary_embedding, rotary_embedding
from tests.test_infer.test_kernels.triton.kernel_utils import (
    mock_alloc_block_table_and_kvcache_v2,
    mock_alloc_block_table_and_kvcache_v3,
    mock_alloc_single_token,
)

inference_ops = InferenceOpsLoader().load()

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
        line_vals=[
            "triton_rotary_emb_func",
            "triton_fused_rotary_emb_func",
            "triton_fused_rotary_emb_func_new_kcache_layout",
            "cuda_rotary_emb_func",
            "cuda_fused_rotary_emb_func",
        ],
        line_names=[
            "triton_rotary_emb_func",
            "triton_fused_rotary_emb_func",
            "triton_fused_rotary_emb_func(new layout)",
            "cuda_rotary_emb_func",
            "cuda_fused_rotary_emb_func",
        ],
        styles=[("red", "-"), ("blue", "-"), ("purple", "-"), ("green", "-"), ("yellow", "-")],
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
    BATCH_SIZE = 16
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
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    new_cache_shape = (BATCH_SIZE * max_num_blocks_per_seq, num_kv_heads, head_dim // x, block_size, x)
    new_k_cache = torch.zeros(size=new_cache_shape, dtype=dtype, device="cuda")

    past_kv_seq_lengths = torch.tensor([SEQ_LEN - 1 for _ in range(BATCH_SIZE)], dtype=torch.int32, device="cuda")
    block_tables = mock_alloc_block_table_and_kvcache_v2(
        k, v, k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_num_blocks_per_seq, block_size
    )
    _ = mock_alloc_block_table_and_kvcache_v3(
        k, v, new_k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_num_blocks_per_seq, block_size
    )
    new_k = torch.randn((BATCH_SIZE, num_kv_heads, head_dim), dtype=dtype, device="cuda")
    new_q = torch.randn_like(new_k)
    new_v = torch.randn_like(new_k)

    mock_alloc_single_token(block_tables, past_kv_seq_lengths, block_size)
    kv_seq_lengths = past_kv_seq_lengths + 1
    block_tables = block_tables.to(device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton_rotary_emb_func":
        fn = lambda: [
            rotary_embedding(new_q, new_k, cos, sin),
            copy_kv_to_blocked_cache(
                new_k, new_v, k_cache, v_cache, kv_lengths=kv_seq_lengths, block_tables=block_tables
            ),
        ]
    elif provider == "triton_fused_rotary_emb_func":
        fn = lambda: decoding_fused_rotary_embedding(
            new_q, new_k, new_v, cos, sin, k_cache, v_cache, block_tables, kv_seq_lengths
        )
    elif provider == "triton_fused_rotary_emb_func_new_kcache_layout":
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        kcache_shape = (BATCH_SIZE * max_num_blocks_per_seq, num_kv_heads, head_dim // x, block_size, x)
        k_cache = torch.zeros(size=kcache_shape, dtype=dtype, device="cuda")
        block_tables = mock_alloc_block_table_and_kvcache_v3(
            k, v, k_cache, v_cache, past_kv_seq_lengths, BATCH_SIZE, max_num_blocks_per_seq, block_size
        )
        mock_alloc_single_token(block_tables, past_kv_seq_lengths, block_size)
        block_tables = block_tables.to(device="cuda")
        fn = lambda: decoding_fused_rotary_embedding(
            new_q, new_k, new_v, cos, sin, k_cache, v_cache, block_tables, kv_seq_lengths, use_new_kcache_layout=True
        )
    elif provider == "cuda_rotary_emb_func":
        fn = lambda: [
            inference_ops.rotary_embedding(new_q, new_k, cos, sin, True),
            inference_ops.decode_kv_cache_memcpy(new_k, new_v, new_k_cache, v_cache, kv_seq_lengths, block_tables),
        ]
    elif provider == "cuda_fused_rotary_emb_func":
        fn = lambda: inference_ops.rotary_embedding_and_cache_copy(
            new_q, new_k, new_v, cos, sin, new_k_cache, v_cache, kv_seq_lengths, block_tables, True
        )
    else:
        raise ValueError("Undefined provider")

    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark_rotary_emb.run(save_path=".", print_data=True)
