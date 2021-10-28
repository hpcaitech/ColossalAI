#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_sequence._operation import RingQK, RingAV
from colossalai.registry import LAYERS


@LAYERS.register_module
class TransformerSelfAttentionRing(nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.

    :param hidden_size: hidden size
    :type hidden_size: int
    :param kv_channels: channels of key/value tensor
    :type kv_channels: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param attention_dropout: dropout probability for attention layer
    :type attention_dropout: float
    """

    def __init__(self,
                 hidden_size,
                 kv_channels,
                 num_attention_heads,
                 attention_dropout,
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_attention_head = projection_size // num_attention_heads

        self.world_size = gpc.get_world_size(ParallelMode.SEQUENCE)

        # Strided linear layer.
        self.query_key_value = nn.Linear(
            hidden_size,
            3 * projection_size,
        )

        # coeff = None
        self.norm_factor = math.sqrt(self.hidden_size)

        # TODO: add apply_query_key_layer_scaling when we have the kernel module
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff

        # TODO: add fused scale mask softmax kernel when we have the kernel module
        # self.scale_mask_softmax = FusedScaleMaskSoftmax(
        #     self.fp16, self.bf16,
        #     self.attn_mask_type,
        #     masked_softmax_fusion,
        #     attention_mask_func,
        #     self.attention_softmax_in_fp32,
        #     coeff)

        self.attention_dropout = nn.Dropout(attention_dropout)

        # Output.
        self.dense = nn.Linear(
            projection_size,
            hidden_size,
            bias=True)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [sq, b, h]

        sub_seq_length, batch_size, hidden_size = hidden_states.size()

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (3 * hn * num_heads)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [sq, b, num_heads, 3 * hn] --> 3 [sq, b, num_heads, hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads,
                                                        3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # split into query, key and value
        last_dim = mixed_x_layer.dim() - 1
        last_dim_value = mixed_x_layer.size()[-1]
        assert last_dim_value % 3 == 0, 'the last dimension is not a multiple of 3, ' \
                                        'cannot be divided into query, key and value'
        partition_size = last_dim_value // 3
        (query_layer, key_layer, value_layer) = torch.split(
            mixed_x_layer, partition_size, dim=last_dim)

        # ===================================
        # Raw attention scores. [b, num_heads, s, s]
        # ===================================

        # [b, num_heads, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0) * self.world_size)

        # [sq, b, num_heads, hn] -> [sq, b * num_heads, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, num_heads, hn] -> [sk, b * num_heads, hn]
        key_layer = key_layer.view(key_layer.size(0),
                                   output_size[0] * output_size[1], -1)

        # [b, sq, sk]
        attention_scores = RingQK.apply(
            # [b * num_heads, sq, hn]
            query_layer.transpose(0, 1).contiguous(),
            key_layer.transpose(0, 1).contiguous(),  # [b * num_heads, sk, hn],
            batch_size,
            self.num_attention_heads,
            sub_seq_length
        )
        attention_scores /= self.norm_factor

        # change view to [b, num_heads, sq, sk]
        attention_scores = attention_scores.view(*output_size)
        attention_scores = attention_scores.unsqueeze(1)

        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.squeeze(1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # with mpu.get_cuda_rng_tracker().fork():
        # TODO: check if a rng tracker is needed
        attention_probs = self.attention_dropout(attention_probs)

        # context layer shape: [b, num_heads, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))
        #
        # # change view [sk, b * num_heads, hn]
        value_layer = value_layer.contiguous().view(value_layer.size(0),
                                                    output_size[0] * output_size[1], -1)

        # # change view [b * num_heads, sq, sk]
        attention_probs = attention_probs.view(attention_probs.size(0) * attention_probs.size(1),
                                               attention_probs.size(2),
                                               attention_probs.size(3))

        # matmul: [b*num_heads, sq, hn]
        # context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        context_layer = RingAV.apply(
            attention_probs,
            value_layer.transpose(0, 1).contiguous(),
            batch_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            sub_seq_length
        )

        # # change view [b, num_heads, sq, hn]
        context_layer = context_layer.view(*output_size)

        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_attention_head * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # context_layer = context_layer.transpose(1, 0).contiguous()
        output = self.dense(context_layer)
        bias = self.dense.bias

        return output, bias
