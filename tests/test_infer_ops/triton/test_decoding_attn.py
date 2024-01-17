import pytest
import torch
from packaging import version

from colossalai.kernel.triton import flash_decoding_attention
from colossalai.utils import get_current_device
from tests.test_infer_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache, torch_attn_ref

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def prepare_padding_mask(kv_lengths: torch.Tensor, bsz: int, max_seq_len: int, device="cuda"):
    padding_mask = torch.zeros((bsz, 1, 1, max_seq_len), dtype=torch.float32, device=device)
    for i in range(bsz):
        cur_seq_len = kv_lengths[i].item()
        assert cur_seq_len <= max_seq_len
        padding_mask[i, :, :, : max_seq_len - cur_seq_len] = float("-inf")
    return padding_mask


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_attn_heads", [16])
@pytest.mark.parametrize("kv_group_num", [1, 2, 16])
@pytest.mark.parametrize("same_context_len", [True, False])
def test_flash_decoding(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_attn_heads: int,
    kv_group_num: int,
    same_context_len: bool,
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    q_len = 1
    head_dim = 128
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    if same_context_len:
        context_lengths = torch.tensor([max_seq_len for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        context_lengths = torch.randint(low=1, high=max_seq_len, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(context_lengths).item()

    q_size = (bsz, q_len, num_attn_heads, head_dim)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    q = q.view(bsz, q_len, num_attn_heads, head_dim)

    kv_size = (num_tokens, 2 * num_kv_heads, head_dim)
    kv = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k, v = torch.split(kv, [num_kv_heads, num_kv_heads], dim=-2)

    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim, block_size)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache(
        k, v, k_cache, v_cache, context_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    block_tables = block_tables.to(device=device)

    max_seq_len = context_lengths.max().item()
    # the maximum block length splitted on kv should be the kv cache block size
    kv_max_split_num = (max_seq_len + block_size - 1) // block_size
    mid_output = torch.empty(
        size=(bsz, num_attn_heads, kv_max_split_num, head_dim), dtype=torch.float32, device=q.device
    )
    mid_output_lse = torch.empty(size=(bsz, num_attn_heads, kv_max_split_num), dtype=torch.float32, device=q.device)
    sm_scale = 1.0 / (head_dim**0.5)
    out_triton = flash_decoding_attention(
        q,
        k_cache,
        v_cache,
        context_lengths,
        block_tables,
        max_seq_len,
        mid_output,
        mid_output_lse,
        block_size=block_size,
        sm_scale=sm_scale,
        kv_group_num=kv_group_num,
    )
    out_triton = out_triton.unsqueeze(1)  # [bsz, 1, num_heads, head_dim]

    # rebuild (batched) kv with padding for torch attention
    # q   [bsz, 1, num_heads, head_dim]
    # k/v [num_tokens, num_kv_heads, head_dim]
    k_torch = torch.zeros((bsz, max_seq_len, num_kv_heads, head_dim), dtype=k.dtype, device=k.device)
    v_torch = torch.zeros_like(k_torch)
    prev_len_sum = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        # mock left-side padding
        k_torch[i, -seq_len:, :, :] = k[prev_len_sum : prev_len_sum + seq_len]
        v_torch[i, -seq_len:, :, :] = v[prev_len_sum : prev_len_sum + seq_len]
        prev_len_sum += seq_len
    # k/v [bsz, max_seq_len, num_kv_heads, head_dim]
    torch_padding_mask = prepare_padding_mask(context_lengths, bsz, k_torch.size(1), q.device)
    out_torch = torch_attn_ref(
        q, k_torch, v_torch, torch_padding_mask, bsz, 1, k_torch.size(1), num_attn_heads, num_kv_heads, head_dim
    )

    assert out_torch.shape == out_triton.shape
    assert torch.allclose(out_torch, out_triton, atol=1e-3, rtol=1e-4)


BATCH = 16
BLOCK_SIZE = 32
SAME_LEN = True
configs = [
    triton.testing.Benchmark(
        x_names=["KV_LEN"],
        x_vals=[2**i for i in range(8, 12)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton"],
        styles=[("red", "-"), ("blue", "-")],
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
    warmup = 10
    rep = 100

    num_attn_heads = 16
    max_num_blocks_per_seq = max(32, triton.cdiv(KV_LEN, block_size))

    num_kv_heads = num_attn_heads // kv_group_num
    assert isinstance(num_kv_heads, int) and num_kv_heads > 0, "Invalid number of kv heads."
    q_len = 1
    head_dim = 128
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    if same_context_len:
        kv_lengths = torch.tensor([KV_LEN for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        kv_lengths = torch.randint(low=1, high=KV_LEN, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(kv_lengths).item()

    q_size = (bsz, q_len, num_attn_heads, head_dim)
    q = torch.empty(size=q_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    kv_size = (num_tokens, 2 * num_kv_heads, head_dim)
    kv = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k, v = torch.split(kv, [num_kv_heads, num_kv_heads], dim=-2)

    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim, block_size)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache(
        k, v, k_cache, v_cache, kv_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    block_tables = block_tables.to(device=device)

    q = q.view(bsz, q_len, num_attn_heads, head_dim)
    max_seq_len = kv_lengths.max().item()  # for random lengths

    if provider == "torch":
        # rebuild (batched) kv with padding for torch attention
        # q   [bsz, 1, num_heads, head_dim]
        # k/v [num_tokens, num_kv_heads, head_dim]
        k_torch = torch.zeros((bsz, max_seq_len, num_kv_heads, head_dim), dtype=k.dtype, device=k.device)
        v_torch = torch.zeros_like(k_torch)
        prev_len_sum = 0
        for i, seq_len in enumerate(kv_lengths.tolist()):
            # mock left-side padding
            k_torch[i, -seq_len:, :, :] = k[prev_len_sum : prev_len_sum + seq_len]
            v_torch[i, -seq_len:, :, :] = v[prev_len_sum : prev_len_sum + seq_len]
            prev_len_sum += seq_len
        # k/v [bsz, max_seq_len, num_kv_heads, head_dim]
        torch_padding_mask = prepare_padding_mask(kv_lengths, bsz, k_torch.size(1), q.device)
        fn = lambda: torch_attn_ref(
            q, k_torch, v_torch, torch_padding_mask, bsz, 1, k_torch.size(1), num_attn_heads, num_kv_heads, head_dim
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    elif provider == "triton":
        # the maximum block length splitted on kv should be the kv cache block size
        kv_max_split_num = (max_seq_len + block_size - 1) // block_size
        mid_output = torch.empty(
            size=(bsz, num_attn_heads, kv_max_split_num, head_dim), dtype=torch.float32, device=q.device
        )
        mid_output_lse = torch.empty(size=(bsz, num_attn_heads, kv_max_split_num), dtype=torch.float32, device=q.device)
        sm_scale = 1.0 / (head_dim**0.5)
        fn = lambda: flash_decoding_attention(
            q,
            k_cache,
            v_cache,
            kv_lengths,
            block_tables,
            max_seq_len,
            mid_output,
            mid_output_lse,
            block_size=block_size,
            sm_scale=sm_scale,
            kv_group_num=kv_group_num,
        ).unsqueeze(1)

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


if __name__ == "__main__":
    # test_flash_decoding(16, 32, 32, 16, 1, True)
    bench_kernel.run(save_path=".", print_data=True)
