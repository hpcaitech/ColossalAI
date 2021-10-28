#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch
from torch import nn as nn, Tensor

from colossalai.nn.layer._common_utils import divide
from colossalai.registry import LAYERS
from ._utils import assert_tesseract_initialization, \
    get_tesseract_dim_dep_from_env
from .layers import Linear2p5D, LayerNorm2p5D
from .._common_utils import ACT2FN


@LAYERS.register_module
class TransformerMLP2p5D(nn.Module):
    """
    MLP will take the input with h hidden state, project it to mlp_ratio * h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    :param in_features: the size of input tensor
    :type in_features: int
    :param mlp_ratio: hidden size of MLP divided by embedding dim, defaults to 4.0
    :type mlp_ratio: int, optional
    :param act_func: activation function, defaults to 'gelu'
    :type act_func: str, optional
    :param dropout_prob: dropout probability, defaults to 0.
    :type dropout_prob: float, optional
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 in_features: int,
                 mlp_ratio: int,
                 act_func: str = 'gelu',
                 dropout_prob: float = 0.,
                 dtype=None,
                 ):
        super().__init__()
        assert_tesseract_initialization()
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()
        self.in_features = in_features

        # Project to h * mlp_ratio.
        self.dense_1 = Linear2p5D(
            in_features,
            mlp_ratio * in_features,
            dtype=dtype
        )

        assert act_func in ACT2FN.keys(), f'Invalid value for argument act_func, ' \
                                          f'activation function can only be {list(ACT2FN.keys())}'
        self.activation_func = ACT2FN[act_func]

        # Project back to h.
        self.dense_2 = Linear2p5D(
            mlp_ratio * in_features,
            in_features,
            dtype=dtype
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm = LayerNorm2p5D(in_features, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        intermediate_output = self.dense_1(x)
        intermediate_output = self.activation_func(intermediate_output)
        output = self.dense_2(intermediate_output)
        output = self.dropout(output)
        output = self.layernorm(x + output)
        return output


@LAYERS.register_module
class TransformerSelfAttention2p5D(nn.Module):
    """Self attention layer for 2.5D parallel Transformer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param attention_dropout_prob: dropout probability for attention layer
    :type attention_dropout_prob: float
    :param hidden_dropout_prob: dropout probability for hidden layer
    :type hidden_dropout_prob: float
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 hidden_dropout_prob,
                 dtype=None,
                 ):
        super().__init__()

        assert_tesseract_initialization()
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()
        self.hidden_size = hidden_size
        self.num_attention_heads = divide(
            num_attention_heads, self.tesseract_dim)  # *
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_key_value = Linear2p5D(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
        )
        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.dense = Linear2p5D(
            hidden_size,
            hidden_size,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layernorm = LayerNorm2p5D(
            hidden_size,
            dtype=dtype)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        query_key_value = self.query_key_value(hidden_states)
        new_qkv_shape = query_key_value.shape[:-1] + \
                        (self.num_attention_heads, 3 * self.attention_head_size)
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))
        query_layer, key_layer, value_layer = torch.chunk(
            query_key_value, 3, dim=-1)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute((0, 2, 1, 3)).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)
        output = self.dropout(output)
        attention_output = self.layernorm(hidden_states + output)

        return attention_output


@LAYERS.register_module
class TransformerLayer2p5D(nn.Module):
    """Transformer layer which contains a self-attention layer and a MLP layer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param act_func: activation function, defaults to 'gelu'
    :type act_func: str, optional
    :param mlp_ratio: hidden size of MLP divided by embedding dim, defaults to 4.0
    :type mlp_ratio: float, optional
    :param attention_dropout_prob: dropout probability for attention layer, defaults to 0.
    :type attention_dropout_prob: float, optional
    :param hidden_dropout_prob: dropout probability for attention layer, defaults to 0.
    :type hidden_dropout_prob: float, optional
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 act_func='gelu',
                 mlp_ratio=4,
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.,
                 dtype=None,
                 ):
        super().__init__()

        self.attention = TransformerSelfAttention2p5D(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            dtype=dtype,
        )
        self.mlp = TransformerMLP2p5D(
            in_features=hidden_size,
            dropout_prob=hidden_dropout_prob,
            act_func=act_func,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        output = self.mlp(attention_output)
        return output
