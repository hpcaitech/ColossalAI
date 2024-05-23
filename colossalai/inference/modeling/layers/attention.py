import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def copy_to_cache(source, cache, lengths, block_tables, type: str = "prefill"):
    """
    Func: copy key/value into key/value cache.

    Args:   key/value(source): shape [bsz,seq_len,num_heads,head_size]
            cache: shape [num_blocks, num_kv_heads, head_size, block_size]
            lengths: key/value lengths
            block_tables
    """
    num_blocks, num_heads, block_size, head_size = cache.shape
    bsz, max_blocks_per_seq = block_tables.shape
    needed_blocks = (lengths + block_size - 1) // block_size

    if type == "prefill":
        for i in range(bsz):
            seq_len = lengths[i]
            block_num = needed_blocks[i]
            token_id = 0
            for block_idx in range(block_num - 1):
                cache[block_tables[i][block_idx]] = source[i][token_id : token_id + block_size].permute(1, 0, 2)
                token_id += block_size
            cache[block_tables[i][block_num - 1], :, : seq_len - token_id, :] = source[i][token_id:seq_len].permute(
                1, 0, 2
            )
    elif type == "decoding":
        assert source.size(1) == 1, "seq_len should be equal to 1 when decoding."
        source = source.squeeze(1)
        slot_idx = (lengths + block_size - 1) % block_size
        for i in range(bsz):
            cache[block_tables[i, needed_blocks[i] - 1], :, slot_idx[i], :] = source[i]

    return cache


def convert_kvcache(cache, lengths, block_tables, pad_id=0):
    """
    Func: convert key/value cache for calculation

    Args:   cache: shape [num_blocks, num_heads, block_size, head_size]
            lengths: key/value length
            block_tables
            pad_id: padded_id
    """
    num_blocks, num_heads, block_size, head_size = cache.shape

    needed_blocks = (lengths + block_size - 1) // block_size
    num_remaing_tokens = lengths % block_size
    num_remaing_tokens[num_remaing_tokens == 0] += block_size
    bsz = block_tables.shape[0]
    seq_len = max(lengths)
    padded_cache = []
    for i in range(bsz):
        _cache = torch.cat(
            (
                cache[block_tables[i][: needed_blocks[i] - 1]].permute((0, 2, 1, 3)).reshape(-1, num_heads, head_size),
                cache[block_tables[i][needed_blocks[i] - 1], :, : num_remaing_tokens[i], :].permute(1, 0, 2),
            ),
            dim=0,
        )
        padding = seq_len - _cache.size(0)
        if padding > 0:
            _cache = F.pad(_cache, (0, 0, 0, 0, 0, padding), value=pad_id)
        padded_cache.append(_cache)
    return torch.stack(padded_cache, dim=0)


class PagedAttention:
    """
    Pure Torch implementation version of paged_attention.
        Holds different types of forward function and useful components.
    """

    @staticmethod
    def pad_and_reshape(tensor, seq_lengths, max_seq_len, num_heads, head_size):
        """
        Transform 1D no_pad tensor into 2D padded tensor with shape [bsz,seq_len,num_heads,head_size]
        """
        bsz = len(seq_lengths)
        padded_tensor = torch.zeros(bsz, max_seq_len, num_heads, head_size, dtype=tensor.dtype)

        token_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            seq_tensor = tensor[token_idx : token_idx + seq_len]
            padded_tensor[i, :seq_len, :, :] = seq_tensor
            token_idx += seq_len
        return padded_tensor

    @staticmethod
    def generate_padding_mask(lengths, max_seq_len):
        range_tensor = torch.arange(max_seq_len).expand(len(lengths), max_seq_len)
        padding_mask = range_tensor < lengths.unsqueeze(1)
        return padding_mask

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int = 1) -> torch.Tensor:
        """
        Essential component for MQA. Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
            Args: hidden_states(batch, num_key_value_heads, seqlen, head_dim)
                  n_rep: times of repeatition.
            Output: hidden_states (batch, num_attention_heads, seqlen, head_dim)
        """
        if n_rep == 1:
            return hidden_states

        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        num_attention_heads = n_rep * num_key_value_heads
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)

        return hidden_states.reshape(batch, num_attention_heads, seq_len, head_dim)

    @staticmethod
    def nopad_context_forward(
        q: torch.Tensor,  # [num_tokens, num_heads, head_size]
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
        v_cache: torch.Tensor,
        context_lengths: torch.Tensor,  # [num_seqs]
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        """
        NOTE: q,k,v are projected and applied rotary embedding, all aligned with triton version.
        """
        # Fisrt, do shape verification
        num_tokens, num_heads, head_size = q.shape
        num_kv_heads = k.shape[-2]

        assert num_heads % num_kv_heads == 0, "num_kv_heads should be divisible by num_heads"
        num_kv_groups = num_heads // num_kv_heads

        block_size = k_cache.size(-2)
        bsz, max_blocks_per_sequence = block_tables.shape
        max_seq_len = max_blocks_per_sequence * block_size
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert context_lengths.shape[0] == block_tables.shape[0]
        shape = (bsz, max_seq_len, num_heads, head_size)
        input_shape = shape[:2]

        q = PagedAttention.pad_and_reshape(
            q, context_lengths, max_seq_len, num_heads, head_size
        )  # bsz,seqlen,num_heads,head_size
        k = PagedAttention.pad_and_reshape(k, context_lengths, max_seq_len, num_heads, head_size)
        v = PagedAttention.pad_and_reshape(v, context_lengths, max_seq_len, num_heads, head_size)

        copy_to_cache(k, k_cache, lengths=context_lengths, block_tables=block_tables)
        copy_to_cache(v, v_cache, lengths=context_lengths, block_tables=block_tables)

        attn_mask = AttentionMaskConverter._make_causal_mask(input_shape, q.dtype, q.device, past_key_values_length=0)
        attn_mask = attn_mask + PagedAttention.generate_padding_mask(context_lengths, max_seq_len)

        q = q.transpose(1, 2)
        k = PagedAttention.repeat_kv(k.transpose(1, 2), num_kv_groups)
        v = PagedAttention.repeat_kv(v.transpose(1, 2), num_kv_groups)

        # position_ids = torch.arange(0, max_seq_len, dtype=torch.long, device=query.device)
        # position_ids = position_ids.unsqueeze(0)
        # cos, sin = self.rotary_emb(value, max_seq_len)
        # query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_size)
        if attn_weights.size() != (bsz, num_heads, max_seq_len, max_seq_len):
            raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,max_seq_len,max_seq_len)}.")

        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (bsz, num_heads, max_seq_len, head_size):
            raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,max_seq_len,head_size)}.")
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, max_seq_len, -1)

        del attn_weights

        return attn_output

    @staticmethod
    def pad_context_forward(
        q: torch.Tensor,  # [batch_size, seq_len, num_heads, head_size]
        k: torch.Tensor,  # [batch_size, seq_len, num_kv_heads, head_size]
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
        v_cache: torch.Tensor,
        context_lengths: torch.Tensor,  # [num_seqs]
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
        attn_mask: torch.Tensor = None,  # [bsz, input_lengths + output_lengths]
    ):
        # Firt, do shape verification
        bsz, seq_len, num_heads, head_size = q.shape
        num_kv_heads = k.shape[-2]
        assert num_heads % num_kv_heads == 0, "num_kv_heads should be divisible by num_heads"
        num_kv_groups = num_heads // num_kv_heads
        block_size = k_cache.size(-2)
        assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]
        block_tables.shape[-1] * block_size

        # Copy kv to memory(rotary embedded)
        copy_to_cache(k, k_cache, lengths=context_lengths, block_tables=block_tables)
        copy_to_cache(v, v_cache, lengths=context_lengths, block_tables=block_tables)

        q = q.transpose(1, 2)
        k = PagedAttention.repeat_kv(k.transpose(1, 2), num_kv_groups)
        v = PagedAttention.repeat_kv(v.transpose(1, 2), num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_size)

        padding_mask = None

        if attn_mask is not None:
            padding_mask = AttentionMaskConverter._expand_mask(attn_mask, q.dtype, seq_len)

        attn_mask = AttentionMaskConverter._make_causal_mask(
            (bsz, seq_len), q.dtype, q.device, past_key_values_length=seq_len - seq_len
        )

        if padding_mask is not None:
            attn_mask = attn_mask.masked_fill(padding_mask.bool(), torch.finfo(q.dtype).min)

        if attn_weights.size() != (bsz, num_heads, seq_len, seq_len):
            raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,seq_len,seq_len)}.")
        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (bsz, num_heads, seq_len, head_size):
            raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,seq_len,head_size)}.")

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)

        return attn_output

    @staticmethod
    def pad_decoding_forward(
        q: torch.Tensor,  # [bsz, 1, num_heads, head_size]
        k: torch.Tensor,  # [bsz, 1, num_kv_heads, head_size]
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
        v_cache: torch.Tensor,
        lengths: torch.Tensor,  # [num_seqs]: input_lengths + output_lengths
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
        attn_mask: torch.Tensor = None,  # [bsz, input_lengths + output_lengths]
    ):
        # Firt, do shape verification.
        bsz, q_length, num_heads, head_size = q.shape

        num_kv_heads = k.shape[-2]
        assert num_heads % num_kv_heads == 0, "num_kv_heads should be divisible by num_heads"
        num_kv_groups = num_heads // num_kv_heads
        seq_len = max(lengths)

        assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]

        copy_to_cache(k, k_cache, lengths=lengths, block_tables=block_tables, type="decoding")
        copy_to_cache(v, v_cache, lengths=lengths, block_tables=block_tables, type="decoding")

        k = convert_kvcache(k_cache, lengths, block_tables)  # bsz, seqlen,
        v = convert_kvcache(v_cache, lengths, block_tables)

        q = q.transpose(1, 2)
        k = PagedAttention.repeat_kv(k.transpose(1, 2), num_kv_groups)
        v = PagedAttention.repeat_kv(v.transpose(1, 2), num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_size)
        if attn_weights.size() != (bsz, num_heads, 1, seq_len):
            raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,1,seq_len)}.")

        padding_mask = None
        if attn_mask is not None:
            padding_mask = AttentionMaskConverter._expand_mask(attn_mask, q.dtype, q_length)

        attn_mask = AttentionMaskConverter._make_causal_mask(
            (bsz, q_length), q.dtype, q.device, past_key_values_length=seq_len - q_length
        )

        if padding_mask is not None:
            attn_mask = attn_mask.masked_fill(padding_mask.bool(), torch.finfo(q.dtype).min)

        attn_weights += attn_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (bsz, num_heads, 1, head_size):
            raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,1,head_size)}.")
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, 1, -1)

        return attn_output

    @staticmethod
    def no_pad_decoding_forward(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, head_size]
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
        v_cache: torch.Tensor,
        lengths: torch.Tensor,  # [num_seqs]: input_lengths + output_lengths
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        return self.pad_decoding_forward(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), k_cache, v_cache, lengths, block_tables
        )
