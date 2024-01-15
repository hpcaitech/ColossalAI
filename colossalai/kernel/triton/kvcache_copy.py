import torch
import triton
import triton.language as tl


# Triton 2.1.0
@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    KV,  # K or V
    KVCache,  # KCache or VCache
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_cacheb,
    stride_cacheh,
    stride_cached,
    stride_cachebs,
    stride_bts,
    stride_btb,
    block_size,
    HEAD_DIM: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    cur_kv_seq_len = tl.load(context_lengths + cur_seq_idx)
    last_bt_block_idx = cur_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = (cur_kv_seq_len % block_size) * stride_cachebs
    offsets_dmodel = tl.arange(0, HEAD_DIM)
    offsets_kv = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel * stride_kd
    kv = tl.load(KV + offsets_kv)
    offsets_kvcache = (
        block_id * stride_cacheb
        + cur_kv_head_idx * stride_cacheh
        + offsets_dmodel * stride_cached
        + offsets_in_last_block
    )
    tl.store(KVCache + offsets_kvcache, kv)
    return


# Used with blocked kv cache.
# Copy k or v to block k/v cache during decoding stage
def copy_kv_to_blocked_cache(
    k: torch.Tensor,  #  [bsz, 1, num_kv_heads, head_dim], k or v during decoding stage
    k_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_dim, block_size], blocked k or v cache (for now, the shapes of them are the same)
    context_lengths: torch.Tensor,  # [bsz], past kv seq len (not incorporating the current kv of length 1)
    block_tables: torch.Tensor,  # [bsz, max_blocks_per_sequence]
):
    assert k.dim() == 4, "Unsupported shape of k (supposed to be used for decoding stage)"
    assert k.size(1) == 1, "Unsupported kv seq len (supposed to be used for decoding stage)"
    assert k.size(-1) == k_cache.size(-2), "Incompatible head dim"
    assert k.dtype == k_cache.dtype, "Expected consistent dtype for tensor and cache."
    bsz, _, num_kv_heads, head_dim = k.shape
    assert context_lengths.shape[0] == block_tables.shape[0] == bsz, (
        f"Got incompatible batch size (number of seqs):\n"
        f"  Conext lengths bsz {context_lengths.shape[0]}, Block tables bsz {block_tables.shape[0]}, "
        f"batch size {bsz}"
    )

    # Modify if the shape of kv cahce is changed.
    block_size = k_cache.size(-1)
    # [bsz, 1, num_kv_heads, head_dim] -> [bsz, num_kv_heads, head_dim]
    k = k.squeeze(dim=1)

    num_warps = 8 if head_dim > 128 else 4

    grid = (bsz, num_kv_heads)
    _copy_to_kvcache_seqlen1_kernel[grid](
        k,
        k_cache,
        block_tables,
        context_lengths,
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
        HEAD_DIM=head_dim,
        num_warps=num_warps,
    )
