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


# Attention calculation adapted from HuggingFace transformers repository
# src/transformers/models/llama/modeling_llama.py
# https://github.com/huggingface/transformers/blob/633215ba58fe5114d8c8d32e415a04600e010701/src/transformers/models/llama/modeling_llama.py#L350
def torch_attn_ref(
    q: torch.Tensor,  # [bsz, seq_len, num_heads, head_dim]
    k: torch.Tensor,  # [bsz, kv_seq_len, num_heads, head_dim]
    v: torch.Tensor,  # [bsz, kv_seq_len, num_heads, head_dim]
    attention_mask: torch.Tensor,  # [bsz, 1, seq_len, kv_seq_len]
    bsz: int,
    seq_len: int,
    kv_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == head_dim
    q = q.view(bsz, seq_len, num_heads, head_dim)
    k = k.view(bsz, kv_seq_len, num_kv_heads, head_dim)
    v = v.view(bsz, kv_seq_len, num_kv_heads, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # repeat kv for GQA and MQA
    # k/v won't change if kv_group_num is 1
    assert num_heads % num_kv_heads == 0, "Number of heads is not multiple of kv heads"
    kv_group_num = num_heads // num_kv_heads
    k = repeat_kv(k, kv_group_num)
    v = repeat_kv(v, kv_group_num)

    qk = torch.matmul(q, k.transpose(2, 3))
    attn_scores = qk / (head_dim**0.5)

    assert attn_scores.shape == (bsz, num_heads, seq_len, kv_seq_len), "Invalid shape of attention scores"
    # for left-side padding
    if attention_mask.size() != (bsz, 1, seq_len, kv_seq_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, seq_len, kv_seq_len)}, but is {attention_mask.size()}"
        )

    attn_scores = attn_scores + attention_mask
    attn_weights = F.softmax(attn_scores.to(dtype=torch.float32), dim=-1).to(dtype=q.dtype)
    out = torch.matmul(attn_weights, v)
    if out.size() != (bsz, num_heads, seq_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, seq_len, head_dim)}, but is" f" {out.size()}"
        )
    out = out.transpose(1, 2).contiguous()
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
):
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


def mock_alloc_single_token(block_tables: torch.Tensor, context_lengths: torch.Tensor, block_size: int):
    """Allocate 1 token on the block table for each seqs in block tables.
    It won't change provided context_lengths
    """

    # consider max_block_id as the last physical block allocated
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
