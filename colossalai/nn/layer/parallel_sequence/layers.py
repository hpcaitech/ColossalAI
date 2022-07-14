#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import colossalai

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_sequence._operation import RingQK, RingAV
from colossalai.registry import LAYERS
from colossalai.kernel.cuda_native.scaled_softmax import AttnMaskType
from colossalai.kernel import FusedScaleMaskSoftmax
from colossalai.context import seed


@LAYERS.register_module
class TransformerSelfAttentionRing(nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.

    Args:
        hidden_size (int): hidden size.
        num_attention_heads (int): number of attention heads.
        attention_dropout (float): dropout probability for attention layer.
        attention_mask_func (:class:`typing.Callable`): Mask function to be applied.
        layer_number (int): number of layers.

    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout,
                 attention_mask_func,
                 layer_number,
                 apply_query_key_layer_scaling: bool = False,
                 convert_fp16_to_fp32_in_softmax: bool = False,
                 attn_mask_type=AttnMaskType.padding,
                 masked_softmax_fusion=True,
                 fp16=False,
                 bf16=False):
        super().__init__()
        self.convert_fp16_to_fp32_in_softmax = convert_fp16_to_fp32_in_softmax
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_mask_func = attention_mask_func
        self.layer_number = layer_number
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attn_mask_type = attn_mask_type
        assert self.layer_number > 0
        self.attention_dropout = attention_dropout

        if self.apply_query_key_layer_scaling:
            self.convert_fp16_to_fp32_in_softmax = True

        assert self.hidden_size % self.num_attention_heads == 0, \
            'hidden size is not divisible by the number of attention heads'

        self.hidden_size_per_attention_head = self.hidden_size // num_attention_heads

        self.world_size = gpc.get_world_size(ParallelMode.SEQUENCE)

        # Strided linear layer.
        self.query_key_value = _Linear(
            hidden_size,
            3 * self.hidden_size,
        )

        self.coeff = None
        self.norm_factor = math.sqrt(self.hidden_size)

        if self.apply_query_key_layer_scaling:
            self.coeff = layer_number
            self.norm_factor *= self.coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(fp16, bf16, self.attn_mask_type, masked_softmax_fusion,
                                                        self.attention_mask_func, self.convert_fp16_to_fp32_in_softmax,
                                                        self.coeff)

        self.attention_dropout = nn.Dropout(attention_dropout)

        # Output.
        self.dense = _Linear(hidden_size, hidden_size, bias=True, skip_bias_add=True)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [sub_seq_len, batch_size, hidden_size]
        # attention_mask: [batch_size, 1, sub_seq_len, seq_len]
        sub_seq_length, batch_size, hidden_size = hidden_states.size()

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads shape change:
        # [sub_seq_len, batch_size, hidden_size] --> [sub_seq_len, batch_size, (3 * head_size * num_heads)]
        mixed_x_layer = self.query_key_value(hidden_states)

        # [sub_seq_len, batch_size, num_heads, 3 * head_size] --> 3 [sub_seq_len, batch_size, num_heads, head_size]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads,
                                                        3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # split into query, key and value
        last_dim = mixed_x_layer.dim() - 1
        last_dim_value = mixed_x_layer.size(-1)
        assert last_dim_value % 3 == 0, 'the last dimension is not a multiple of 3, ' \
                                        'cannot be divided into query, key and value'
        partition_size = last_dim_value // 3
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, partition_size, dim=last_dim)

        # attention scores: [batch_size, num_heads, sub_seq_len, seq_len]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0),
                       key_layer.size(0) * self.world_size)

        # [sub_seq_len, batch_size, num_heads, head_size] -> [sub_seq_len, batch_size * num_heads, head_size]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sub_seq_len, batch_size, num_heads, head_size] -> [sub_seq_len, batch_size * num_heads, head_size]
        key_layer = key_layer.view(key_layer.size(0), output_size[0] * output_size[1], -1)

        # attention_scores: [batch_size * num_heads, sub_seq_len, seq_len]
        attention_scores = RingQK.apply(
            query_layer.transpose(0, 1).contiguous(),    # [batch_size * num_heads, sub_seq_len, head_size]
            key_layer.transpose(0, 1).contiguous(),    # [batch_size * num_heads, sub_seq_len, head_size],
            batch_size,
            self.num_attention_heads,
            sub_seq_length)

        attention_scores /= self.norm_factor

        # change view to [batch_size, num_heads, sub_seq_len, seq_len]
        attention_scores = attention_scores.view(*output_size)

        # change shape to [batch_size, num_heads, sub_seq_len, seq_len]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_probs)

        # context layer shape: [batch_size, num_heads, sub_seq_len, head_size]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sub_seq_len, batch_size * num_heads, head_size]
        value_layer = value_layer.contiguous().view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # # change view [b * num_heads, sub_seq_len, seq_len]
        attention_probs = attention_probs.view(
            attention_probs.size(0) * attention_probs.size(1), attention_probs.size(2), attention_probs.size(3))

        # matmul: [batch_size * num_heads, sub_seq_len, head_size]
        context_layer = RingAV.apply(attention_probs,
                                     value_layer.transpose(0, 1).contiguous(), batch_size, self.num_attention_heads,
                                     self.hidden_size_per_attention_head, sub_seq_length)

        # change view [batch_size, num_heads, sub_seq_len, head_size]
        context_layer = context_layer.view(*output_size)

        # [batch_size, num_heads, sub_seq_len, head_size] -> [sub_seq_len, batch_size, num_heads, head_size]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sub_seq_len, batch_size, num_heads, head_size] -> [sub_seq_len, batch_size, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_attention_head *
                                                               self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output, bias = self.dense(context_layer)

        return output, bias

    def __repr__(self):
        return f'TransformerSelfAttentionRing(apply_query_key_layer_scaling={self.apply_query_key_layer_scaling}, ' \
            f'layer_number={self.layer_number}, hidden_size:{self.hidden_size}, attention_dropout={self.attention_dropout}, ' \
            f'attn_mask_type={self.attn_mask_type}, num_attention_heads={self.num_attention_heads}, ' \
            f'hidden_size_per_attention_head={self.hidden_size_per_attention_head}, coeff={self.coeff}, norm_factor={self.norm_factor}, ' \
            f'convert_fp16_to_fp32_in_softmax={self.convert_fp16_to_fp32_in_softmax})'


class _Linear(nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, skip_bias_add=False):
        super(_Linear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        self.weight = Parameter(torch.empty(
            self.output_size,
            self.input_size,
        ))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(input_, self.weight, bias)

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output

    def __repr__(self):
        return f'Linear(in_features={self.input_size}, out_features={self.output_size}, ' + \
            f'bias={self.bias is not None}, skip_bias_add={self.skip_bias_add})'
