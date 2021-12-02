#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Tuple

from colossalai.context import seed, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from .._common_utils import divide, ACT2FN
from .._parallel_utilities import reduce_grad, reduce_input, gather_forward_split_backward, \
    split_forward_gather_backward
from ..base_layer import ParallelLayer
from .layers import Linear1D_Col, Linear1D_Row
from .layers import MixedFusedLayerNorm1D as LayerNorm1D

@LAYERS.register_module
class TransformerMLP1D(ParallelLayer):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self,
                 in_features: int,
                 mlp_ratio: int = 4.0,
                 act_func: str = 'gelu',
                 dropout_prob: float = 0.,
                 dtype=None,
                 skip_bias_add: bool = False
                 ):
        super(TransformerMLP1D, self).__init__()
        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.skip_bias_add = skip_bias_add
        # Project to h * mlp_ratio.
        self.dense_1 = Linear1D_Col(
            self.in_features,
            int(self.mlp_ratio * self.in_features),
            bias=not skip_bias_add,
            dtype=dtype,
            gather_output = False,
        )

        assert act_func in ACT2FN.keys(), f'Invalid value for argument act_func, ' \
                                          f'activation function can only be {list(ACT2FN.keys())}'
        self.activation_func = ACT2FN[act_func]

        # Project back to h.
        self.dense_2 = Linear1D_Row(
            int(self.mlp_ratio * self.in_features),
            self.in_features,
            bias=not skip_bias_add,
            dtype=dtype,
            parallel_input = True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        # self.layernorm = LayerNorm1D(in_features, dtype=dtype)
        self.layernorm = nn.LayerNorm(in_features, dtype=dtype)
    def forward(self, x):
        if self.skip_bias_add:
            intermediate_output, _ = self.dense_1(x)
        else:
            intermediate_output = self.dense_1(x)

        intermediate_output = self.activation_func(intermediate_output)

        if self.skip_bias_add:
            output, _ = self.dense_2(intermediate_output)
        else:
            output = self.dense_2(intermediate_output)

        with seed(ParallelMode.TENSOR):
            output = self.dropout(output)
        output = self.layernorm(x + output)
        return output

@LAYERS.register_module
class TransformerSelfAttention1D(ParallelLayer):
    """Self attention layer for 1D parallel Transformer

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
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 ):

        super().__init__()

        self.hidden_size = hidden_size

        self.num_attention_heads = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)

        self.query_key_value = Linear1D_Col(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
        )
        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.dense = Linear1D_Row(
            hidden_size,
            hidden_size,
            dtype=dtype,
            parallel_input=True,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # need to re-enable torch grad to enable fused optimization.
        # self.layernorm = LayerNorm1D(
        #     hidden_size,
        #     dtype=dtype)
        self.layernorm = nn.LayerNorm(
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
        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute((0, 2, 1, 3)).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)
        with seed(ParallelMode.TENSOR):
            output = self.dropout(output)
        attention_output = self.layernorm(hidden_states + output)

        return attention_output

@LAYERS.register_module
class TransformerLayer1D(ParallelLayer):
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
                 hidden_size: int,
                 num_attention_heads: int,
                 act_func: str = 'gelu',
                 mlp_ratio: float = 4.0,
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.,
                 dtype=None,
                 ):
        super().__init__()

        self.attention = TransformerSelfAttention1D(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            dtype=dtype,
        )
        self.mlp = TransformerMLP1D(
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
