import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def forward_fn():
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        # replace dropout process with added DropoutForParallelInput layer
        # origin code:
        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs = self.dropout_layer(attn_weights)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs

    return forward


def get_sam_flash_attention_forward():
    from transformers.models.sam.modeling_sam import SamAttention

    try:
        from xformers.ops import memory_efficient_attention as me_attention
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")

    def _separate_heads(hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states

    def _recombine_heads(hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_tokens, n_heads, c_per_head = hidden_states.shape
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(
        self: SamAttention, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None
    ) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = _separate_heads(query, self.num_attention_heads)
        key = _separate_heads(key, self.num_attention_heads)
        value = _separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        bias = None
        if attention_similarity is not None:
            bias = attention_similarity

        scale = 1.0 / math.sqrt(c_per_head)
        out = me_attention(query, key, value, attn_bias=bias, scale=scale)

        out = _recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out

    return forward


def get_sam_vision_flash_attention_forward():
    from transformers.models.sam.modeling_sam import SamVisionAttention

    try:
        from xformers.ops import memory_efficient_attention as me_attention
    except:
        raise ImportError("Error: xformers module is not installed. Please install it to use flash attention.")

    def add_decomposed_rel_pos(
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """

        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, nHead, dim = query.shape
        reshaped_query = query.transpose(1, 2).reshape(batch_size * nHead, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        rel_pos = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        rel_pos = rel_pos.reshape(batch_size, nHead, query_height * query_width, key_height * key_width)
        return rel_pos

    def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def forward(self: SamVisionAttention, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 1, 3, 4)
        )

        query, key, value = qkv.reshape(3, batch_size, height * width, self.num_attention_heads, -1).unbind(0)

        rel_pos = None
        if self.use_rel_pos:
            rel_pos = add_decomposed_rel_pos(query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width))

        attn_output = me_attention(query, key, value, attn_bias=rel_pos, p=self.dropout, scale=self.scale)

        attn_output = attn_output.reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        outputs = (attn_output, None)

        return outputs

    return forward
