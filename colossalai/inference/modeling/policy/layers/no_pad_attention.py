from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


class NoPadPagedAttention(nn.Module):
    """
    Pure Torch implementation version of paged_attention.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float, sliding_window: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.sliding_window = sliding_window
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_size,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

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

    def forward(
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
        bsz, max_seq_len = block_tables.shape
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert context_lengths.shape[0] == block_tables.shape[0]
        shape = (bsz, max_seq_len, num_heads, head_size)
        query = self.pad_and_reshape(q, context_lengths, max_seq_len, num_heads, head_size).transpose(1, 2)
        key = self.pad_and_reshape(k, context_lengths, max_seq_len, num_heads, head_size).transpose(1.2)
        value = self.pad_and_reshape(v, context_lengths, max_seq_len, num_heads, head_size).transpose(1, 2)

        attn_mask = AttentionMaskConverter._make_causal_mask(shape, q.dtype, q.device, past_key_values_length=0)
        self.generate_padding_mask(context_lengths, max_seq_len)

        cos, sin = self.rotary_emb(value, max_seq_len)
        query, value = apply_rotary_pos_emb(query, key)
