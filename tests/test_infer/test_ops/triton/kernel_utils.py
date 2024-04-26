from typing import Tuple

import torch
from torch.nn import functional as F


# This function is adapted from src/transformers/models/llama/modeling_llama.py
# in huggingface transformers repository
# https://github.com/huggingface/transformers/blob/3b7675b2b844b02d4821b827871a21ad16dd446c/src/transformers/models/llama/modeling_llama.py#L273
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (bsz, num_key_value_heads, seq_len, head_dim) to (bsz, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


def create_attention_mask(kv_lengths: torch.Tensor, bsz: int, q_len: int, kv_len: int, device="cuda"):
    assert q_len <= kv_len

    causal_mask = torch.full((q_len, q_len), fill_value=float("-inf"), device=device).triu(diagonal=1)

    padding_mask = torch.zeros((bsz, 1, q_len, kv_len), dtype=torch.float32, device=device)
    for i in range(bsz):
        cur_seq_len = kv_lengths[i].item()
        assert cur_seq_len <= kv_len
        padding_mask[i, :, :, : kv_len - cur_seq_len] = float("-inf")

    padding_mask[:, :, -q_len:, -q_len:] += causal_mask

    return padding_mask


# Attention calculation adapted from HuggingFace transformers repository
# src/transformers/models/llama/modeling_llama.py
# https://github.com/huggingface/transformers/blob/633215ba58fe5114d8c8d32e415a04600e010701/src/transformers/models/llama/modeling_llama.py#L350
def torch_attn_ref(
    q: torch.Tensor,  # [bsz, num_heads, q_len, head_dim]
    k: torch.Tensor,  # [bsz, num_heads, kv_len, head_dim]
    v: torch.Tensor,  # [bsz, num_heads, kv_len, head_dim]
    attention_mask: torch.Tensor,  # [bsz, 1, q_len, kv_len]
    bsz: int,
    q_len: int,
    kv_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == head_dim

    # repeat kv for GQA and MQA
    # k/v won't change if kv_group_num is 1
    assert num_heads % num_kv_heads == 0, "Number of heads is not multiple of kv heads"
    kv_group_num = num_heads // num_kv_heads
    k = repeat_kv(k, kv_group_num)
    v = repeat_kv(v, kv_group_num)

    qk = torch.matmul(q, k.transpose(2, 3))
    attn_scores = qk / (head_dim**0.5)

    assert attn_scores.shape == (bsz, num_heads, q_len, kv_len), "Invalid shape of attention scores"
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    attn_weights = F.softmax(attn_scores.to(dtype=torch.float32), dim=-1).to(dtype=q.dtype)
    out = torch.matmul(attn_weights, v)
    if out.size() != (bsz, num_heads, q_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is" f" {out.size()}"
        )
    out = out.transpose(1, 2).contiguous()
    out = out.view(-1, out.size(-2), out.size(-1))
    # out [bsz * q_len, num_heads, head_dim]
    return out


def mock_alloc_block_table_and_kvcache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    num_seqs: int,
    max_num_blocks_per_seq: int,
    block_size: int,
) -> torch.Tensor:
    """Allocate block tables based on provided context lengths; and copy KV to blocked KV Cache."""
    block_id = 0
    block_tables = torch.full(size=(num_seqs, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
    num_tokens_processed = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        right_bound = (seq_len + block_size - 1) // block_size  # open bound
        block_tables[i, :right_bound] = torch.arange(block_id, block_id + right_bound, dtype=torch.int32)
        # Manually fill kv caches by copying from k and v
        for i in range(right_bound):
            if i == right_bound - 1:
                allocated_locs = seq_len % block_size or block_size
            else:
                allocated_locs = block_size
            k_block = k[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            v_block = v[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            k_cache[block_id, :, :, :allocated_locs] = k_block
            v_cache[block_id, :, :, :allocated_locs] = v_block

            num_tokens_processed += allocated_locs
            block_id += 1

    return block_tables


def mock_alloc_block_table_and_kvcache_v2(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    num_seqs: int,
    max_num_blocks_per_seq: int,
    block_size: int,
) -> torch.Tensor:
    """Allocate block tables based on provided context lengths; and copy KV to blocked KV Cache."""
    block_id = 0
    block_tables = torch.full(size=(num_seqs, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
    num_tokens_processed = 0
    for i, seq_len in enumerate(context_lengths.tolist()):
        right_bound = (seq_len + block_size - 1) // block_size  # open bound
        block_tables[i, :right_bound] = torch.arange(block_id, block_id + right_bound, dtype=torch.int32)
        # Manually fill kv caches by copying from k and v
        for i in range(right_bound):
            if i == right_bound - 1:
                allocated_locs = seq_len % block_size or block_size
            else:
                allocated_locs = block_size
            k_block = k[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 0, 2)
            v_block = v[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 0, 2)
            k_cache[block_id, :, :allocated_locs, :] = k_block
            v_cache[block_id, :, :allocated_locs, :] = v_block

            num_tokens_processed += allocated_locs
            block_id += 1

    return block_tables


def mock_alloc_block_table_and_kvcache_v3(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    num_seqs: int,
    max_num_blocks_per_seq: int,
    block_size: int,
) -> torch.Tensor:
    """Allocate block tables based on provided context lengths; and copy KV to blocked KV Cache."""
    block_id = 0
    block_tables = torch.full(size=(num_seqs, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
    num_tokens_processed = 0

    _, num_kv_heads, head_dim = k.shape

    x = 16 // torch.tensor([], dtype=k.dtype).element_size()

    for i, seq_len in enumerate(context_lengths.tolist()):
        right_bound = (seq_len + block_size - 1) // block_size  # open bound
        block_tables[i, :right_bound] = torch.arange(block_id, block_id + right_bound, dtype=torch.int32)
        # Manually fill kv caches by copying from k and v
        for i in range(right_bound):
            if i == right_bound - 1:
                allocated_locs = seq_len % block_size or block_size
            else:
                allocated_locs = block_size
            # [block_size, num_kv_heads, head_dim/x, x]->[num_kv_heads, head_dim/x, block_size,x]
            k_block = (
                k[num_tokens_processed : num_tokens_processed + allocated_locs, :, :]
                .reshape(allocated_locs, num_kv_heads, head_dim // x, x)
                .permute(1, 2, 0, 3)
            )
            v_block = v[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 0, 2)
            k_cache[block_id, :, :, :allocated_locs, :] = k_block
            v_cache[block_id, :, :allocated_locs, :] = v_block

            num_tokens_processed += allocated_locs
            block_id += 1

    return block_tables


def mock_alloc_block_table_and_kvcache_vllm(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    num_seqs: int,
    max_num_blocks_per_seq: int,
    block_size: int,
) -> torch.Tensor:
    """Allocate block tables based on provided context lengths; and copy KV to blocked KV Cache."""
    block_id = 0
    block_tables = torch.full(size=(num_seqs, max_num_blocks_per_seq), fill_value=-1, dtype=torch.int32)
    num_tokens_processed = 0

    _, num_kv_heads, head_dim = k.shape

    x = 16 // torch.tensor([], dtype=k.dtype).element_size()

    for i, seq_len in enumerate(context_lengths.tolist()):
        right_bound = (seq_len + block_size - 1) // block_size  # open bound
        block_tables[i, :right_bound] = torch.arange(block_id, block_id + right_bound, dtype=torch.int32)
        # Manually fill kv caches by copying from k and v
        for i in range(right_bound):
            if i == right_bound - 1:
                allocated_locs = seq_len % block_size or block_size
            else:
                allocated_locs = block_size
            # [block_size, num_kv_heads, head_dim/x, x]->[num_kv_heads, head_dim/x, block_size,x]
            k_block = (
                k[num_tokens_processed : num_tokens_processed + allocated_locs, :, :]
                .reshape(allocated_locs, num_kv_heads, head_dim // x, x)
                .permute(1, 2, 0, 3)
            )
            # [block_size, num_kv_heads, head_dim]->[num_kv_heads, head_dim, block_size]
            v_block = v[num_tokens_processed : num_tokens_processed + allocated_locs, :, :].permute(1, 2, 0)
            k_cache[block_id, :, :, :allocated_locs, :] = k_block
            v_cache[block_id, :, :, :allocated_locs] = v_block

            num_tokens_processed += allocated_locs
            block_id += 1

    return block_tables


def mock_alloc_single_token(block_tables: torch.Tensor, context_lengths: torch.Tensor, block_size: int) -> None:
    # Allocate 1 token on the block table for each seqs in block tables.
    # It won't change provided context_lengths.
    # Consider max_block_id as the last physical block allocated
    # NOTE It assumes all the blocks preceding this block have been allocated
    max_block_id = torch.max(block_tables).item()
    # the indices on each block table representing the cache block to be allocated one more token
    alloc_local_block_indices = context_lengths // block_size
    # offsets of the token to be allocated on the target block (for each seq)
    alloc_block_offsets = context_lengths % block_size

    require_new_block = alloc_block_offsets == 0
    new_block_ids = torch.arange(
        max_block_id + 1,
        max_block_id + 1 + require_new_block.sum(),
        dtype=block_tables.dtype,
        device=block_tables.device,
    )

    if new_block_ids.numel():
        new_block_alloc_local_indices = alloc_local_block_indices[require_new_block]
        block_tables[require_new_block, new_block_alloc_local_indices] = new_block_ids


def generate_caches_and_block_tables(
    k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype=torch.float16, device="cuda"
) -> Tuple[torch.Tensor, ...]:
    # Mock generation of k/v blocked caches and block tables from providied kv unpad and seq lengths
    # k_unpad/v_unpad [num_total_tokens, num_kv_heads, head_dim]
    _, num_kv_heads, head_dim = k_unpad.shape
    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim, block_size)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache(
        k_unpad, v_unpad, k_cache, v_cache, kv_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    return k_cache, v_cache, block_tables


def generate_caches_and_block_tables_v2(
    k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype=torch.float16, device="cuda"
) -> Tuple[torch.Tensor, ...]:
    # Mock generation of k/v blocked caches and block tables from providied kv unpad and seq lengths
    # k_unpad/v_unpad [num_total_tokens, num_kv_heads, head_dim]
    _, num_kv_heads, head_dim = k_unpad.shape
    cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, block_size, head_dim)
    k_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache_v2(
        k_unpad, v_unpad, k_cache, v_cache, kv_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    return k_cache, v_cache, block_tables


def generate_caches_and_block_tables_v3(
    k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype=torch.float16, device="cuda"
) -> Tuple[torch.Tensor, ...]:
    # Mock generation of k/v blocked caches and block tables from providied kv unpad and seq lengths
    # k_unpad/v_unpad [num_total_tokens, num_kv_heads, head_dim]
    _, num_kv_heads, head_dim = k_unpad.shape

    x = 16 // torch.tensor([], dtype=dtype).element_size()

    k_cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim // x, block_size, x)
    v_cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, block_size, head_dim)
    k_cache = torch.zeros(size=k_cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=v_cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache_v3(
        k_unpad, v_unpad, k_cache, v_cache, kv_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    return k_cache, v_cache, block_tables


def generate_caches_and_block_tables_vllm(
    k_unpad, v_unpad, kv_lengths, bsz, max_num_blocks_per_seq, block_size, dtype=torch.float16, device="cuda"
) -> Tuple[torch.Tensor, ...]:
    # Mock generation of k/v blocked caches and block tables from providied kv unpad and seq lengths
    # k_unpad/v_unpad [num_total_tokens, num_kv_heads, head_dim]
    _, num_kv_heads, head_dim = k_unpad.shape

    x = 16 // torch.tensor([], dtype=dtype).element_size()

    k_cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim // x, block_size, x)
    v_cache_shape = (bsz * max_num_blocks_per_seq, num_kv_heads, head_dim, block_size)
    k_cache = torch.zeros(size=k_cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros(size=v_cache_shape, dtype=dtype, device=device)
    # Mock allocation on block tables as well as blocked kv caches
    block_tables = mock_alloc_block_table_and_kvcache_vllm(
        k_unpad, v_unpad, k_cache, v_cache, kv_lengths, bsz, max_num_blocks_per_seq, block_size
    )
    return k_cache, v_cache, block_tables


def convert_kv_unpad_to_padded(
    k_unpad: torch.Tensor, kv_seq_lengths: torch.Tensor, bsz: int, max_seq_len: int
) -> torch.Tensor:
    # Rebuild (batched) k/v with padding to be used by torch attention
    # input k_unpad/v_unpad [num_total_tokens, num_kv_heads, head_dim]
    # returns k/v padded    [bsz, num_kv_heads, max_seq_len, head_dim]
    _, num_kv_heads, head_dim = k_unpad.shape
    k_torch = torch.zeros((bsz, max_seq_len, num_kv_heads, head_dim), dtype=k_unpad.dtype, device=k_unpad.device)
    prev_len_sum = 0
    for i, seq_len in enumerate(kv_seq_lengths.tolist()):
        # left-side padding
        k_torch[i, -seq_len:, :, :] = k_unpad[prev_len_sum : prev_len_sum + seq_len]
        prev_len_sum += seq_len
    k_torch = k_torch.transpose(1, 2)
    return k_torch
