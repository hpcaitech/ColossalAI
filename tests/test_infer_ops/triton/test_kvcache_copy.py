import pytest
import torch
from packaging import version

from colossalai.inference.modeling.layers.attention import copy_to_cache
from colossalai.kernel.triton import copy_kv_to_blocked_cache
from colossalai.utils import get_current_device
from tests.test_infer_ops.triton.kernel_utils import mock_alloc_block_table_and_kvcache, mock_alloc_single_token

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")


def prepare_data(
    bsz,
    num_kv_heads,
    head_dim,
    block_size,
    max_num_blocks_per_seq,
    same_context_len,
    max_seq_len,
    device,
    dtype=torch.float16,
):
    if same_context_len:
        # past_kv_seq_lengths in this test records the previous kv seq len
        # (not incorporating the current input whose seq len is 1)
        past_kv_seq_lengths = torch.tensor([max_seq_len - 1 for _ in range(bsz)], dtype=torch.int32, device=device)
    else:
        past_kv_seq_lengths = torch.randint(low=1, high=max_seq_len - 1, size=(bsz,), dtype=torch.int32, device=device)
    num_tokens = torch.sum(past_kv_seq_lengths).item()

    kv_size = (num_tokens, 2 * num_kv_heads, head_dim)
    kv = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k, v = torch.split(kv, [num_kv_heads, num_kv_heads], dim=-2)

    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim, block_size)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache(
        k, v, k_cache, v_cache, past_kv_seq_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    block_tables = block_tables.to(device=device)

    new_k = torch.randn((bsz, 1, num_kv_heads, head_dim), dtype=dtype, device=device)
    # mock allocating blocks for the new k/v and update block tables
    mock_alloc_single_token(block_tables, past_kv_seq_lengths, block_size)

    # kv seq len = past kv seq len + seq len (1 during decoding stage)
    kv_seq_lengths = past_kv_seq_lengths + 1

    return new_k, k_cache, kv_seq_lengths, block_tables


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [4, 7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [8, 32])
@pytest.mark.parametrize("num_kv_heads", [16])
@pytest.mark.parametrize("same_context_len", [True, False])
def test_copy_kv_to_caches(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_kv_heads: int,
    same_context_len: bool,
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    head_dim = 128
    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    new_k, k_cache, kv_seq_lengths, block_tables = prepare_data(
        bsz,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks_per_seq,
        same_context_len,
        max_seq_len,
        device=device,
        dtype=dtype,
    )
    copy_kv_to_blocked_cache(new_k, k_cache, kv_seq_lengths, block_tables)

    for seq_i in range(bsz):
        ki = new_k[seq_i]
        ki = ki.squeeze()
        past_kv_seq_len = kv_seq_lengths[seq_i] - 1
        target_block_id = block_tables[seq_i, past_kv_seq_len // block_size]
        offsets_in_block = past_kv_seq_len % block_size
        target = k_cache[target_block_id, :, :, offsets_in_block]
        orig = new_k[seq_i].squeeze(dim=0)
        assert torch.equal(orig, target)


BATCH = 16
configs = [
    triton.testing.Benchmark(
        x_names=["KV_SEQ_LEN"],
        x_vals=[2**i for i in range(8, 13)],
        line_arg="provider",
        line_vals=["torch_copy_func", "triton_copy_func"],
        line_names=["torch_copy_func", "triton_copy_func"],
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"kvcache_copy_decoding_stage-batch-{BATCH}",
        args={"bsz": BATCH, "block_size": 16, "max_seq_len": 8192, "num_kv_heads": 16, "same_context_len": True},
    )
]


@triton.testing.perf_report(configs)
def benchmark_kvcache_copy(
    provider: str,
    bsz: int,
    block_size: int,
    max_seq_len: int,
    KV_SEQ_LEN: int,  # maximum past kv length (unequal context lens in batch) or past kv len (equal context lens)
    num_kv_heads: int,
    same_context_len: bool,
):
    warmup = 10
    rep = 100

    head_dim = 128
    dtype = torch.float16
    device = get_current_device()

    assert KV_SEQ_LEN <= max_seq_len, "Assigned maximum kv length must be smaller or equal to maximum seq len"

    new_k, k_cache, context_lengths, block_tables = prepare_data(
        bsz,
        num_kv_heads,
        head_dim,
        block_size,
        max_seq_len // block_size,
        same_context_len,
        KV_SEQ_LEN,
        device=device,
        dtype=dtype,
    )

    if provider == "torch_copy_func":
        fn = lambda: copy_to_cache(new_k, k_cache, lengths=context_lengths, block_tables=block_tables, type="decoding")
    elif provider == "triton_copy_func":
        fn = lambda: copy_kv_to_blocked_cache(new_k, k_cache, context_lengths, block_tables)
    else:
        raise ValueError("Undefined provider.")

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    test_copy_kv_to_caches(4, 32, 8, 16, True)
    # benchmark_kvcache_copy.run(save_path=".", print_data=True)
