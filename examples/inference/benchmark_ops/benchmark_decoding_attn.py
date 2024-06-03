import torch

from colossalai.kernel.triton import flash_decoding_attention
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.triton.kernel_utils import (
    convert_kv_unpad_to_padded,
    create_attention_mask,
    generate_caches_and_block_tables_v2,
    generate_caches_and_block_tables_v3,
    torch_attn_ref,
)
from tests.test_infer.test_kernels.triton.test_decoding_attn import prepare_data

try:
    import triton  # noqa

except ImportError:
    print("please install triton from https://github.com/openai/triton")

Q_LEN = 1
HEAD_DIM = 128
BATCH = 16
BLOCK_SIZE = 32
SAME_LEN = True
WARM_UPS = 10
REPS = 100
configs = [
    triton.testing.Benchmark(
        x_names=["KV_LEN"],
        x_vals=[2**i for i in range(8, 14)],
        # x_vals=[x for x in range(256, 8192, 256)],
        line_arg="provider",
        line_vals=["torch", "triton", "triton_new_kcache_layout"],
        line_names=["Torch", "Triton", "Triton New KCache Layout"],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms",
        plot_name=f"decoding-block_size-{BLOCK_SIZE}-batch{BATCH}",
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
    block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    q, k_unpad, v_unpad, kv_lengths = prepare_data(
        bsz, num_attn_heads, num_kv_heads, HEAD_DIM, same_context_len, Q_LEN, max_seq_len, dtype, device
    )
    max_seq_len_in_b = kv_lengths.max().item()  # for random lengths
    # the maximum block length splitted on kv should be the kv cache block size
    kv_max_split_num = (max_seq_len_in_b + block_size - 1) // block_size
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    output = torch.empty((bsz, num_attn_heads, HEAD_DIM), dtype=dtype, device=device)
    mid_output = torch.empty(
        size=(bsz, num_attn_heads, kv_max_split_num, HEAD_DIM), dtype=torch.float32, device=q.device
    )
    mid_output_lse = torch.empty(size=(bsz, num_attn_heads, kv_max_split_num), dtype=torch.float32, device=q.device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        k_torch = convert_kv_unpad_to_padded(k_unpad, kv_lengths, bsz, max_seq_len_in_b)
        v_torch = convert_kv_unpad_to_padded(v_unpad, kv_lengths, bsz, max_seq_len_in_b)
        torch_padding_mask = create_attention_mask(kv_lengths, bsz, Q_LEN, max_seq_len_in_b, q.device)
        fn = lambda: torch_attn_ref(
            q,
            k_torch,
            v_torch,
            torch_padding_mask,
            bsz,
            Q_LEN,
            max_seq_len_in_b,
            num_attn_heads,
            num_kv_heads,
            HEAD_DIM,
        )
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)
    elif provider == "triton":
        k_cache, v_cache, block_tables = generate_caches_and_block_tables_v2(
            k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
        )
        block_tables = block_tables.to(device=device)
        fn = lambda: flash_decoding_attention(
            # Here we use q.squeeze(2) because we hide the q_len dimension (which is equivalent to 1),
            # refer to attention forward in modeling.
            q.squeeze(2),
            k_cache,
            v_cache,
            kv_lengths,
            block_tables,
            block_size,
            max_seq_len_in_b,
            output,
            mid_output,
            mid_output_lse,
            sm_scale=sm_scale,
            kv_group_num=kv_group_num,
        )  # [bsz, 1, num_heads, head_dim]
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)
    elif provider == "triton_new_kcache_layout":
        k_cache, v_cache, block_tables = generate_caches_and_block_tables_v3(
            k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype, device
        )
        block_tables = block_tables.to(device=device)
        fn = lambda: flash_decoding_attention(
            # Here we use q.squeeze(2) because we hide the q_len dimension (which is equivalent to 1),
            # refer to attention forward in modeling.
            q.squeeze(2),
            k_cache,
            v_cache,
            kv_lengths,
            block_tables,
            block_size,
            max_seq_len_in_b,
            output,
            mid_output,
            mid_output_lse,
            sm_scale=sm_scale,
            kv_group_num=kv_group_num,
            use_new_kcache_layout=True,
        )
        ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    bench_kernel.run(save_path=".", print_data=True)
