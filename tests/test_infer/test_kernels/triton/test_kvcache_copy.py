import pytest
import torch
from packaging import version

from colossalai.kernel.triton import copy_k_to_blocked_cache, copy_kv_to_blocked_cache
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.triton.kernel_utils import (
    generate_caches_and_block_tables_v2,
    generate_caches_and_block_tables_v3,
    mock_alloc_single_token,
)

try:
    import triton  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

TRITON_CUDA_SUPPORT = version.parse(torch.version.cuda) > version.parse("11.4")

HEAD_DIM = 32


def prepare_data(
    bsz,
    num_kv_heads,
    head_dim,
    block_size,
    max_num_blocks_per_seq,
    same_context_len,
    max_seq_len,
    n=1,
    device="cuda",
    dtype=torch.float16,
    use_new_kcache_layout=False,
):
    assert max_seq_len > n, "max_seq_len must be greater than n"

    past_kv_seq_lengths = (
        torch.tensor([max_seq_len - n for _ in range(bsz)], dtype=torch.int32, device=device)
        if same_context_len
        else torch.randint(low=1, high=max_seq_len - n, size=(bsz,), dtype=torch.int32, device=device)
    )
    num_tokens = torch.sum(past_kv_seq_lengths).item()

    kv_size = (num_tokens, 2 * num_kv_heads, head_dim)
    kv_unpad = torch.empty(size=kv_size, dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k_unpad, v_unpad = torch.split(kv_unpad, [num_kv_heads, num_kv_heads], dim=-2)

    if use_new_kcache_layout:
        k_cache, v_cache, block_tables = generate_caches_and_block_tables_v3(
            k_unpad, v_unpad, past_kv_seq_lengths, bsz, max_num_blocks_per_seq, block_size, dtype=dtype, device=device
        )
    else:
        k_cache, v_cache, block_tables = generate_caches_and_block_tables_v2(
            k_unpad, v_unpad, past_kv_seq_lengths, bsz, max_num_blocks_per_seq, block_size, dtype=dtype, device=device
        )
    block_tables = block_tables.to(device=device)

    new_k = torch.randn((bsz, n, num_kv_heads, head_dim), dtype=dtype, device=device)
    new_v = torch.randn((bsz, n, num_kv_heads, head_dim), dtype=dtype, device=device)
    # mock allocating blocks for the new k/v and update block tables
    for _ in range(n):
        mock_alloc_single_token(block_tables, past_kv_seq_lengths, block_size)
        past_kv_seq_lengths += 1

    return new_k, new_v, k_cache, v_cache, past_kv_seq_lengths, block_tables


@pytest.mark.skipif(not (HAS_TRITON and TRITON_CUDA_SUPPORT), reason="requires triton")
@pytest.mark.parametrize("bsz", [7, 32])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16])
@pytest.mark.parametrize("num_kv_heads", [16])
@pytest.mark.parametrize("same_context_len", [True, False])
@pytest.mark.parametrize("n_tokens", [1, 5])
@pytest.mark.parametrize("use_new_kcache_layout", [True, False])
def test_copy_kv_to_caches(
    bsz: int,
    block_size: int,
    max_num_blocks_per_seq: int,
    num_kv_heads: int,
    same_context_len: bool,
    n_tokens: int,
    use_new_kcache_layout: bool,
):
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    max_seq_len = block_size * max_num_blocks_per_seq
    dtype = torch.float16
    device = get_current_device()

    new_k, new_v, k_cache, v_cache, kv_seq_lengths, block_tables = prepare_data(
        bsz,
        num_kv_heads,
        HEAD_DIM,
        block_size,
        max_num_blocks_per_seq,
        same_context_len,
        max_seq_len,
        n_tokens,
        device=device,
        dtype=dtype,
        use_new_kcache_layout=use_new_kcache_layout,
    )
    k_source = new_k.view(-1, new_k.size(-2), new_k.size(-1))
    v_source = new_v.view(-1, new_v.size(-2), new_v.size(-1))
    k_cache_copy = k_cache.detach().clone()
    past_kv_seq_lengths = kv_seq_lengths - n_tokens
    target_block_ids = block_tables[range(0, block_tables.size(0)), past_kv_seq_lengths // block_size]
    offsets_in_block = past_kv_seq_lengths % block_size

    # Copy k (or v) to k (or v) cache
    copy_k_to_blocked_cache(
        new_k, k_cache, kv_seq_lengths, block_tables, n=n_tokens, use_new_kcache_layout=use_new_kcache_layout
    )
    # Reshape target k from k cache to compare if matching with original tensor
    # Mainly to handle cases of n_tokens > 1
    k_target = []
    for i in range(bsz):
        block_table = block_tables[i]
        curr_kv_len = past_kv_seq_lengths[i].item()
        offset = offsets_in_block[i].item()
        tokens_left = n_tokens
        while tokens_left > 0:
            tokens_to_fill = min(block_size - offset, tokens_left)
            curr_block_id = block_table[curr_kv_len // block_size]
            if use_new_kcache_layout:
                k_target.append(k_cache[curr_block_id, :, :, offset : offset + tokens_to_fill, :])
            else:
                k_target.append(k_cache[curr_block_id, :, offset : offset + tokens_to_fill, :])
            curr_kv_len += tokens_to_fill
            tokens_left -= tokens_to_fill
            offset = 0
    if use_new_kcache_layout:
        k_target = torch.concat(k_target, dim=2).permute(2, 0, 1, 3).contiguous()
        k_target = k_target.reshape(bsz * n_tokens, num_kv_heads, HEAD_DIM)
    else:
        k_target = torch.concat(k_target, dim=1).transpose(0, 1).contiguous()  # [bsz * n, num_kv_heads, head_dim]
    assert k_target.shape == k_source.shape
    assert torch.equal(k_target, k_source)

    if n_tokens == 1:
        # Copy k and v to k/v caches
        k_cache = k_cache_copy
        copy_kv_to_blocked_cache(
            new_k, new_v, k_cache, v_cache, kv_seq_lengths, block_tables, use_new_kcache_layout=use_new_kcache_layout
        )

        if use_new_kcache_layout:
            k_target = k_cache[target_block_ids, :, :, offsets_in_block, :]
            k_target = k_target.contiguous().reshape(bsz * n_tokens, num_kv_heads, HEAD_DIM)
        else:
            k_target = k_cache[target_block_ids, :, offsets_in_block, :]
        assert k_target.shape == k_source.shape
        assert torch.equal(k_target, k_source)
        v_target = v_cache[target_block_ids, :, offsets_in_block, :]
        assert v_target.shape == v_source.shape
        assert torch.equal(v_target, v_source)


if __name__ == "__main__":
    test_copy_kv_to_caches(4, 32, 8, 16, True, n_tokens=1)
