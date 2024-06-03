import torch

from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import flash_decoding_attention
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.triton.kernel_utils import (
    generate_caches_and_block_tables_v2,
    generate_caches_and_block_tables_v3,
    generate_caches_and_block_tables_vllm,
)

try:
    import triton  # noqa
except ImportError:
    print("please install triton from https://github.com/openai/triton")

inference_ops = InferenceOpsLoader().load()

# Triton benchmark plot attributions
configs = [
    triton.testing.Benchmark(
        x_names=["MAX_NUM_BLOCKS_PER_SEQ"],
        x_vals=[2**i for i in range(2, 8)],
        line_arg="provider",
        line_vals=[
            "vllm_paged_decoding_attention",
            "triton_flash_decoding_attention",
            "cuda_flash_decoding_attention",
        ],
        line_names=[
            "vllm_paged_decoding_attention",
            "triton_flash_decoding_attention",
            "cuda_flash_decoding_attention",
        ],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-")],
        ylabel="ms",
        plot_name=f"FlashDecodingAttention benchmarking results",
        args={"BATCH_SIZE": 16, "BLOCK_SIZE": 32, "HEAD_SIZE": 128, "KV_GROUP_NUM": 2},
    )
]


def prepare_data(
    BATCH_SIZE: int,
    HEAD_SIZE: int,
    NUM_ATTN_HEADS: int,
    NUM_KV_HEADS: int,
    MAX_SEQ_LEN: int,
    dtype=torch.float16,
    device="cuda",
):
    # Use the provided maximum sequence length for each sequence when testing with teh same context length,
    # otherwise generate random context lengths.
    # returns
    #   q [BATCH_SIZE, NUM_ATTN_HEADS, HEAD_SIZE]
    #   k_unpad/v_unpad [num_tokens, NUM_KV_HEADS, HEAD_SIZE]
    kv_lengths = torch.randint(low=1, high=MAX_SEQ_LEN, size=(BATCH_SIZE,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(kv_lengths).item()

    q_size = (BATCH_SIZE, 1, NUM_ATTN_HEADS, HEAD_SIZE)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5).transpose(1, 2)
    kv_size = (num_tokens, 2 * NUM_KV_HEADS, HEAD_SIZE)
    kv_unpad = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k_unpad, v_unpad = torch.split(kv_unpad, [NUM_KV_HEADS, NUM_KV_HEADS], dim=-2)

    return q, k_unpad, v_unpad, kv_lengths


@triton.testing.perf_report(configs)
def benchmark_flash_decoding_attention(
    provider: str,
    BATCH_SIZE: int,
    BLOCK_SIZE: int,
    MAX_NUM_BLOCKS_PER_SEQ: int,
    HEAD_SIZE: int,
    KV_GROUP_NUM: int,
):
    try:
        from vllm._C import ops as vllm_ops
    except ImportError:
        raise ImportError("Please install vllm from https://github.com/vllm-project/vllm")

    warmup = 10
    rep = 1000

    dtype = torch.float16

    NUM_ATTN_HEADS = 16

    NUM_KV_HEADS = NUM_ATTN_HEADS // KV_GROUP_NUM
    assert isinstance(NUM_KV_HEADS, int) and NUM_KV_HEADS > 0, "Invalid number of kv heads."
    MAX_SEQ_LEN = BLOCK_SIZE * MAX_NUM_BLOCKS_PER_SEQ
    device = get_current_device()

    q, k_unpad, v_unpad, kv_seq_lengths = prepare_data(
        BATCH_SIZE, HEAD_SIZE, NUM_ATTN_HEADS, NUM_KV_HEADS, MAX_SEQ_LEN, dtype, device
    )

    triton_k_cache, triton_v_cache, _ = generate_caches_and_block_tables_v2(
        k_unpad, v_unpad, kv_seq_lengths, BATCH_SIZE, MAX_NUM_BLOCKS_PER_SEQ, BLOCK_SIZE, dtype, device
    )

    k_cache, v_cache, block_tables = generate_caches_and_block_tables_v3(
        k_unpad, v_unpad, kv_seq_lengths, BATCH_SIZE, MAX_NUM_BLOCKS_PER_SEQ, BLOCK_SIZE, dtype, device
    )

    vllm_k_cache, vllm_v_cache, _ = generate_caches_and_block_tables_vllm(
        k_unpad, v_unpad, kv_seq_lengths, BATCH_SIZE, MAX_NUM_BLOCKS_PER_SEQ, BLOCK_SIZE, dtype, device
    )

    block_tables = block_tables.to(device=device)
    max_seq_len_across_batch = kv_seq_lengths.max().item()
    kv_max_split_num = (max_seq_len_across_batch + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty((BATCH_SIZE, NUM_ATTN_HEADS, HEAD_SIZE), dtype=dtype, device=device)
    sm_scale = 1.0 / (HEAD_SIZE**0.5)
    alibi_slopes = None
    kv_scale = 1.0

    mid_output = torch.empty(
        size=(BATCH_SIZE, NUM_ATTN_HEADS, kv_max_split_num, HEAD_SIZE), dtype=torch.float32, device=device
    )
    mid_output_lse = torch.empty(
        size=(BATCH_SIZE, NUM_ATTN_HEADS, kv_max_split_num), dtype=torch.float32, device=device
    )
    exp_sums = torch.empty(size=(BATCH_SIZE, NUM_ATTN_HEADS, kv_max_split_num), dtype=torch.float32, device=device)
    max_logits = torch.empty(size=(BATCH_SIZE, NUM_ATTN_HEADS, kv_max_split_num), dtype=torch.float32, device=device)

    if provider == "vllm_paged_decoding_attention":
        alibi_slopes = None
        fn = lambda: vllm_ops.paged_attention_v1(
            output,
            q.squeeze(2),
            vllm_k_cache,
            vllm_v_cache,
            NUM_KV_HEADS,
            sm_scale,
            block_tables,
            kv_seq_lengths,
            BLOCK_SIZE,
            max_seq_len_across_batch,
            alibi_slopes,
            "auto",
            kv_scale,
        )
    elif provider == "triton_flash_decoding_attention":
        fn = lambda: flash_decoding_attention(
            q.squeeze(2),
            triton_k_cache,
            triton_v_cache,
            kv_seq_lengths,
            block_tables,
            BLOCK_SIZE,
            max_seq_len_across_batch,
            output,
            mid_output,
            mid_output_lse,
            sm_scale=sm_scale,
            kv_group_num=KV_GROUP_NUM,
        )  # [bsz, 1, num_heads, head_dim]
    elif provider == "cuda_flash_decoding_attention":
        fn = lambda: inference_ops.flash_decoding_attention(
            output,
            q.squeeze(2),
            k_cache,
            v_cache,
            kv_seq_lengths,
            block_tables,
            BLOCK_SIZE,
            max_seq_len_across_batch,
            mid_output,
            exp_sums,
            max_logits,
            alibi_slopes,
            sm_scale,
        )
    else:
        raise ValueError("Undefined provider.")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


if __name__ == "__main__":
    benchmark_flash_decoding_attention.run(save_path=".", print_data=True)
