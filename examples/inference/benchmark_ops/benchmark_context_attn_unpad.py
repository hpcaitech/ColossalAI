import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from colossalai.inference.modeling.layers.attention import PagedAttention
from colossalai.kernel.triton import context_attention_unpadded
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.triton.kernel_utils import generate_caches_and_block_tables_v2, torch_attn_ref

try:
    import triton  # noqa

except ImportError:
    print("please install triton from https://github.com/openai/triton")

HEAD_DIM = 32
BATCH = 16
BLOCK_SIZE = 32
SAME_LEN = True
WARM_UPS = 10
REPS = 100
configs = [
    triton.testing.Benchmark(
        x_names=["KV_LEN"],
        x_vals=[2**i for i in range(8, 13)],
        # x_vals=[x for x in range(256, 8192, 256)],
        line_arg="provider",
        line_vals=["torch", "triton", "triton_new_klayout"],
        line_names=["Torch", "Triton", "Triton_new_klayout"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"context_attn-block_size-{BLOCK_SIZE}-batch{BATCH}",
        args={"bsz": BATCH, "block_size": BLOCK_SIZE, "same_context_len": SAME_LEN, "kv_group_num": 1},
    )
]


@triton.testing.perf_report(configs)
def bench_kernel(
    bsz,
    KV_LEN,
    provider,
    block_size: int,
    kv_group_num: int,
    same_context_len: bool,
):
    num_attn_heads = 16
    max_num_blocks_per_seq = triton.cdiv(KV_LEN, block_size)
    max_seq_len = block_size * max_num_blocks_per_seq

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    dtype = torch.float16
    device = get_current_device()

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

    qkv_size = (num_tokens, num_attn_heads + 2 * num_kv_heads, HEAD_DIM)
    qkv_unpad = torch.empty(size=qkv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    q_unpad, k_unpad, v_unpad = torch.split(qkv_unpad, [num_attn_heads, num_kv_heads, num_kv_heads], dim=-2)
    q_unpad = q_unpad.contiguous()
    k_cache_ref, v_cache_ref, block_tables = generate_caches_and_block_tables_v2(
        k_unpad, v_unpad, context_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
    )
    block_tables = block_tables.to(device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        q_padded = PagedAttention.pad_and_reshape(q_unpad, context_lengths, max_seq_len, num_attn_heads, HEAD_DIM)
        k_padded = PagedAttention.pad_and_reshape(k_unpad, context_lengths, max_seq_len, num_kv_heads, HEAD_DIM)
        v_padded = PagedAttention.pad_and_reshape(v_unpad, context_lengths, max_seq_len, num_kv_heads, HEAD_DIM)
        q_padded, k_padded, v_padded = (
            q_padded.to(device=device),
            k_padded.to(device=device),
            v_padded.to(device=device),
        )
        q_padded = q_padded.transpose(1, 2)
        k_padded = PagedAttention.repeat_kv(k_padded.transpose(1, 2), kv_group_num)
        v_padded = PagedAttention.repeat_kv(v_padded.transpose(1, 2), kv_group_num)
        # This benchmark ignores the padding mask. *Only* use the-same-length inputs for benchmarkings
        attn_mask = AttentionMaskConverter._make_causal_mask(
            (bsz, max_seq_len), q_padded.dtype, q_padded.device, past_key_values_length=0
        )
        attn_mask = attn_mask.to(device=q_padded.device)
        fn = lambda: torch_attn_ref(
            q_padded,
            k_padded,
            v_padded,
            attn_mask,
            bsz,
            max_seq_len,
            max_seq_len,
            num_attn_heads,
            num_kv_heads,
            HEAD_DIM,
        )
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)
    elif provider == "triton":
        k_cache_triton = torch.zeros_like(k_cache_ref)
        v_cache_triton = torch.zeros_like(v_cache_ref)
        fn = lambda: context_attention_unpadded(
            q_unpad, k_unpad, v_unpad, k_cache_triton, v_cache_triton, context_lengths, block_tables, block_size
        )
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)
    elif provider == "triton_new_klayout":
        # NOTE New kcache layout (num_blocks, num_kv_heads, head_dim // x, block_size, x)
        # to be applied around the cuda and triton kernels.
        # Here we want to make sure it does not cause downgrade in performance.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        k_cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, HEAD_DIM // x, block_size, x)
        k_cache_triton = torch.zeros(size=k_cache_shape, dtype=dtype, device=device)
        v_cache_triton = torch.zeros_like(v_cache_ref)
        fn = lambda: context_attention_unpadded(
            q_unpad,
            k_unpad,
            v_unpad,
            k_cache_triton,
            v_cache_triton,
            context_lengths,
            block_tables,
            block_size,
            use_new_kcache_layout=True,
        )
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    bench_kernel.run(save_path=".", print_data=True)
