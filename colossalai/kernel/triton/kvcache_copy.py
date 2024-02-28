import torch
import triton
import triton.language as tl


# Triton 2.1.0
@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    KV,  # K or V
    KVCache,  # KCache or VCache
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_cacheb,
    stride_cacheh,
    stride_cachebs,
    stride_cached,
    stride_bts,
    stride_btb,
    block_size,
    n,
    HEAD_DIM: tl.constexpr,
):
    cur_token_idx = tl.program_id(0)
    cur_seq_idx = cur_token_idx // n
    cur_token_shift = cur_token_idx - (n * (cur_seq_idx + 1))
    # cur_token_shift = cur_token_idx - n * cur_seq_idx
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) + cur_token_shift
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offset_last_block = past_kv_seq_len % block_size
    offsets_dmodel = tl.arange(0, HEAD_DIM)
    offsets_kv = cur_token_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel * stride_kd
    kv = tl.load(KV + offsets_kv)
    offsets_kvcache = (
        block_id * stride_cacheb
        + cur_kv_head_idx * stride_cacheh
        + offset_last_block * stride_cachebs
        + offsets_dmodel * stride_cached
    )
    tl.store(KVCache + offsets_kvcache, kv)
    return


# Triton 2.1.0
@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K,  # K
    V,  # V
    KCache,  # KCache
    VCache,  # VCache
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_cachekb,
    stride_cachekh,
    stride_cachekbs,
    stride_cachekd,
    stride_cachevb,
    stride_cachevh,
    stride_cachevbs,
    stride_cachevd,
    stride_bts,
    stride_btb,
    block_size,
    HEAD_DIM: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size
    offsets_dmodel = tl.arange(0, HEAD_DIM)
    offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel * stride_kd
    offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel * stride_vd

    k = tl.load(K + offsets_k)
    v = tl.load(V + offsets_v)

    offsets_kcache = (
        block_id * stride_cachekb
        + cur_kv_head_idx * stride_cachekh
        + offsets_in_last_block * stride_cachekbs
        + offsets_dmodel * stride_cachekd
    )
    offsets_vcache = (
        block_id * stride_cachevb
        + cur_kv_head_idx * stride_cachevh
        + offsets_in_last_block * stride_cachevbs
        + offsets_dmodel * stride_cachevd
    )

    tl.store(KCache + offsets_kcache, k)
    tl.store(VCache + offsets_vcache, v)
    return


def copy_k_to_blocked_cache(
    k: torch.Tensor, k_cache: torch.Tensor, kv_lengths: torch.Tensor, block_tables: torch.Tensor, n: int = 1
):
    """
    Copy keys or values to the blocked key/value cache during decoding stage.

    Args:
        k (torch.Tensor): [bsz, 1, num_kv_heads, head_dim]/[bsz, num_kv_heads, head_dim] - Keys or values during decoding with seq len 1.
            [bsz * n, num_kv_heads, head_dim] - Keys or values with seq len n
        k_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_dim] - Blocked key or value cache.
        kv_lengths (torch.Tensor): [bsz] - Past key/value sequence lengths plus current sequence length for each sequence.
        block_tables (torch.Tensor): [bsz, max_blocks_per_sequence] - Block tables for each sequence.
        n (int): Number of tokens to copy for each sequence. Default to 1.
    """
    assert k.size(-1) == k_cache.size(-1), "Incompatible head dim"
    assert k.dtype == k_cache.dtype, "Expected consistent dtype for tensor and cache."

    k = k.reshape(-1, k.size(-2), k.size(-1)) if k.dim() == 4 else k
    assert k.dim() == 3, f"Invalid k dim {k.dim()}"
    bsz, num_kv_heads, head_dim = k.shape
    # NOTE when n > 1, the shape of k is [bsz * n, num_kv_heads, head_dim]
    if n > 1:
        assert bsz % n == 0, "Each sequence should have the same number of tokens to be copied"
        bsz = bsz // n

    assert kv_lengths.shape[0] == block_tables.shape[0] == bsz, (
        f"Got incompatible batch size (number of seqs):\n"
        f"  Past kv sequence lengths bsz {kv_lengths.shape[0]}; "
        f" block tables bsz {block_tables.shape[0]}, input k batch size {bsz}"
    )

    # Modify if the shape of kv cahce is changed.
    block_size = k_cache.size(-2)

    num_warps = 8 if head_dim > 128 else 4

    grid = (bsz * n, num_kv_heads)
    _copy_to_kcache_seqlen_n_kernel[grid](
        k,
        k_cache,
        block_tables,
        kv_lengths,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        block_size,
        n=n,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
    )


def copy_kv_to_blocked_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_lengths: torch.Tensor,
    block_tables: torch.Tensor,
):
    """
    Copy keys or values to the blocked key/value cache during decoding stage.

    Args:
        k (torch.Tensor): [bsz, 1, num_kv_heads, head_dim]/[bsz, num_kv_heads, head_dim] - Keys during decoding with seq len 1.
        v (torch.Tensor): [bsz, 1, num_kv_heads, head_dim]/[bsz, num_kv_heads, head_dim] - Values during decoding with seq len 1.
        k_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_dim] - Blocked key cache.
        v_cache (torch.Tensor): [num_blocks, num_kv_heads, block_size, head_dim] - Blocked value cache.
        kv_lengths (torch.Tensor): [bsz] - Past key/value sequence lengths plus current sequence length for each sequence.
        block_tables (torch.Tensor): [bsz, max_blocks_per_sequence] - Block tables for each sequence.
    """
    assert k.size(-1) == k_cache.size(-1), "Incompatible head dim"
    assert k.dtype == k_cache.dtype, "Expected consistent dtype for tensor and cache."
    k = k.squeeze(1) if k.dim() == 4 else k
    assert k.dim() == 3, f"Incompatible k dim {k.dim()}"

    assert v.size(-1) == v_cache.size(-1), "Incompatible head dim"
    assert v.dtype == v_cache.dtype, "Expected consistent dtype for tensor and cache."
    v = v.squeeze(1) if v.dim() == 4 else v
    assert v.dim() == 3, f"Incompatible v dim {v.dim()}"

    bsz, num_kv_heads, head_dim = k.shape

    assert kv_lengths.shape[0] == block_tables.shape[0] == bsz, (
        f"Got incompatible batch size (number of seqs):\n"
        f"  Past kv sequence lengths bsz {kv_lengths.shape[0]}; "
        f" block tables bsz {block_tables.shape[0]}, input k batch size {bsz}"
    )

    # Modify if the shape of kv cahce is changed.
    block_size = k_cache.size(-2)

    num_warps = 8 if head_dim > 128 else 4
    grid = (bsz, num_kv_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        kv_lengths,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        block_tables.stride(0),
        block_tables.stride(1),
        block_size,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
    )
