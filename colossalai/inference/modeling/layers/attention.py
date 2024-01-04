import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


def copy_to_cache(source, cache, lengths, block_tables, type: str = "prefill"):
    """
    Func: copy key/value into key/value cache.

    Args:   key/value(source): shape [bsz,seq_len,num_heads,head_size]
            cache: shape [num_blocks, num_heads, head_size, block_size]
            lengths: key/value lengths
            block_tables
    """
    num_blocks, num_heads, head_size, block_size = cache.shape
    bsz, max_seq_len = block_tables.shape
    needed_blocks = (lengths + block_size - 1) // block_size

    if type == "prefill":
        for i in range(bsz):
            seq_len = lengths[i]
            block_num = needed_blocks[i]
            token_id = 0
            for block_idx in range(block_num - 1):
                cache[block_tables[i][block_idx]] = source[i][token_id : token_id + block_size].permute(1, 2, 0)
                token_id += block_size
            cache[block_tables[i][block_num - 1]] = source[i][token_id:seq_len].permute(1, 2, 0)
    elif type == "decoding":
        assert len(source[0]) == 1, "seq_len should be equal to 1 when decoding."
        source = source.squeeze(1)
        slot_idx = (lengths + block_size - 1) % block_size
        for i in range(bsz):
            cache[block_tables[i, needed_blocks[i] - 1], :, :, slot_idx[i]] = source[i].permute(0, 1)

    return cache


def convert_kvcache(source, cache, lengths, block_tables):
    """
    Func: convert key/value cache for calculation

    Args:   key/value(source): shape [bsz, 1, num_heads, head_size]
            cache: shape [num_blocks, num_heads, head_size, block_size]
            lengths: key/value length
            block_tables
    """
    num_blocks, num_heads, head_size, block_size = cache.shape

    needed_blocks = (lengths + block_size - 1) // block_size
    num_remaing_tokens = (lengths - 1) % block_size
    bsz = block_tables.shape[0]
    seq_len = max(lengths)
    padded_cache = []
    for i in range(bsz):
        _cache = torch.cat(
            (
                cache[block_tables[i][: needed_blocks[i] - 1]].permute((3, 0, 1, 2)).reshape(-1, num_heads, head_size),
                cache[block_tables[i][needed_blocks[i] - 1], :, :, : num_remaing_tokens[i]].permute(2, 1, 0),
            ),
            dim=0,
        )
        concat_cache = torch.cat((_cache, source[i]), dim=0)
        padding = seq_len - concat_cache.size(0)
        if padding > 0:
            concat_cache = F.pad(concat_cache, (0, 0, 0, 0, 0, 1))
        padded_cache.append(concat_cache)

    return torch.stack(padded_cache, dim=0)


class PagedAttention(nn.Module):
    """
    Pure Torch implementation version of paged_attention.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float = 1.0, sliding_window: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.sliding_window = sliding_window
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(self.head_size)

    def pad_and_reshape(self, tensor, seq_lengths, max_seq_len, num_heads, head_size):
        bsz = len(seq_lengths)
        padded_tensor = torch.zeros(bsz, max_seq_len, num_heads, head_size)

        token_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            seq_tensor = tensor[token_idx : token_idx + seq_len]
            padded_tensor[i, :seq_len, :, :] = seq_tensor
            token_idx += seq_len
        return padded_tensor

    def generate_padding_mask(self, lengths, max_seq_len):
        range_tensor = torch.arange(max_seq_len).expand(len(lengths), max_seq_len)
        padding_mask = range_tensor < lengths.unsqueeze(1)
        return padding_mask

    def nopad_context_forward(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, head_size]
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
        v_cache: torch.Tensor,
        context_lengths: torch.Tensor,  # [num_seqs]
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        num_tokens, num_heads, head_size = q.shape
        block_size = k_cache.shape[-1]
        bsz, max_blocks_per_sequence = block_tables.shape
        max_seq_len = max_blocks_per_sequence * block_size
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert context_lengths.shape[0] == block_tables.shape[0]
        shape = (bsz, max_seq_len, num_heads, head_size)
        input_shape = shape[:2]
        query = self.pad_and_reshape(q, context_lengths, max_seq_len, num_heads, head_size).transpose(1, 2)
        key = self.pad_and_reshape(k, context_lengths, max_seq_len, num_heads, head_size).transpose(1, 2)
        value = self.pad_and_reshape(v, context_lengths, max_seq_len, num_heads, head_size).transpose(1, 2)

        attn_mask = AttentionMaskConverter._make_causal_mask(input_shape, q.dtype, q.device, past_key_values_length=0)
        self.generate_padding_mask(context_lengths, max_seq_len)

        position_ids = torch.arange(0, max_seq_len, dtype=torch.long, device=query.device)
        position_ids = position_ids.unsqueeze(0)

        cos, sin = self.rotary_emb(value, max_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        copy_to_cache(key.transpose(1, 2), k_cache, lengths=context_lengths, block_tables=block_tables)
        copy_to_cache(value.transpose(1, 2), v_cache, lengths=context_lengths, block_tables=block_tables)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_size)

        if attn_weights.size() != (bsz, num_heads, max_seq_len, max_seq_len):
            raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,max_seq_len,max_seq_len)}.")

        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        # attn_weights = nn.functional.dropout(attn_weights,p=self.attention_dropout,training=False) maybe useless
        attn_output = torch.matmul(attn_weights, value)

        if attn_output.size() != (bsz, num_heads, max_seq_len, head_size):
            raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,max_seq_len,head_size)}.")
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, max_seq_len, -1)

        return attn_output

    def pad_context_forward(
        self,
        q: torch.Tensor,  # [batch_size, seq_len, num_heads, head_size]
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
        v_cache: torch.Tensor,
        context_lengths: torch.Tensor,  # [num_seqs]
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        bsz, seq_len, num_heads, head_size = q.shape
        block_size = k_cache.shape[-1]
        assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]
        block_tables.shape[-1] * block_size
        shape = (bsz, seq_len, num_heads, head_size)
        input_shape = shape[:2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=q.device)
        position_ids = position_ids.unsqueeze(0)
        cos, sin = self.rotary_emb(v, seq_len)
        query, key = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        copy_to_cache(key.transpose(1, 2), k_cache, lengths=context_lengths, block_tables=block_tables)
        copy_to_cache(v.transpose(1, 2), v_cache, lengths=context_lengths, block_tables=block_tables)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_size)
        attn_mask = AttentionMaskConverter._make_causal_mask(input_shape, q.dtype, q.device, past_key_values_length=0)
        self.generate_padding_mask(context_lengths, seq_len)

        if attn_weights.size() != (bsz, num_heads, seq_len, seq_len):
            raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,seq_len,seq_len)}.")
        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # attn_weights = nn.functional.dropout(attn_weights,p=self.attention_dropout,training=False) maybe useless
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (bsz, num_heads, seq_len, head_size):
            raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,seq_len,head_size)}.")

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)

        return attn_output

    def pad_decoding_forward(
        self,
        q: torch.Tensor,  # [bsz, 1, num_heads, head_size]
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, head_size, block_size]
        v_cache: torch.Tensor,
        lengths: torch.Tensor,  # [num_seqs]: input_lengths + output_lengths
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        bsz, _, num_heads, head_size = q.shape
        block_size = k_cache.shape[-1]
        seq_len = max(lengths)

        assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]
        max_seq_len = block_tables.shape[-1] * block_size
        attn_mask = AttentionMaskConverter._make_causal_mask(
            q.shape[:2], q.dtype, q.device, past_key_values_length=seq_len - 1
        )
        self.generate_padding_mask(lengths, max_seq_len)
        cos, sin = self.rotary_emb(v, max_seq_len)

        position_ids = lengths - 1
        position_ids = position_ids.unsqueeze(1)

        query, key = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2)

        copy_to_cache(key, k_cache, lengths=lengths, block_tables=block_tables, type="decoding")
        copy_to_cache(v, v_cache, lengths=lengths, block_tables=block_tables, type="decoding")

        key = convert_kvcache(key, k_cache, lengths, block_tables)  # bsz, seqlen,
        value = convert_kvcache(v, v_cache, lengths, block_tables)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_size)
        if attn_weights.size() != (bsz, num_heads, 1, seq_len):
            raise ValueError(f"Got wrong attn_weights, should be in shape {(bsz,num_heads,1,seq_len)}.")

        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        # attn_weights = nn.functional.dropout(attn_weights,p=self.attention_dropout,training=False) maybe useless
        attn_output = torch.matmul(attn_weights, value)

        if attn_output.size() != (bsz, num_heads, 1, head_size):
            raise ValueError(f"Got wrong attn_output, should be in shape {(bsz,num_heads,1,head_size)}.")
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, 1, -1)

        return attn_output

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
