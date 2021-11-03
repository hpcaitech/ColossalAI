import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from colossalai.constants import (INPUT_GROUP_3D, OUTPUT_GROUP_3D,
                                  WEIGHT_GROUP_3D)
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.nn.init import init_bias_, init_weight_
from colossalai.registry import LAYERS
from colossalai.nn.init import init_bias_, init_weight_
from colossalai.utils import checkpoint, get_current_device
from torch import Tensor, dtype, nn

from .._common_utils import ACT2FN, divide, set_tensor_parallel_attribute_by_partition, to_2tuple
from ._utils import (get_depth_from_env, get_last_group,
                     get_parallel_mode_from_env)
from .layers import Classifier3D, Linear3D, PatchEmbedding3D


@LAYERS.register_module
class ViTPatchEmbedding3D(nn.Module):
    """ 3D Image to Patch Embedding

    :param img_size: iamge size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param in_chans: number of channels of input image
    :type in_chans: int
    :param embed_size: dimension of embedding
    :type embed_size: int
    :param drop_prob: dropout probability
    :type drop_prob: float
    :param flatten: whether to flatten output tensor, defaults to True
    :type flatten: bool, optional
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 drop_prob: float,
                 dtype: dtype = None,
                 flatten: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        init_weight = 'torch'
        init_bias = 'torch'
        if init_method == 'jax':
            init_weight = 'jax_embed'
            init_bias = 'zero'

        self.patch_embed = PatchEmbedding3D(
            img_size,
            patch_size,
            in_chans,
            embed_size,
            dtype=dtype,
            flatten=flatten,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        with seed(ParallelMode.TENSOR):
            x = self.dropout(x)
        return x


@LAYERS.register_module
class ViTSelfAttention3D(nn.Module):
    """Self-attention layer for 3D parallel Vision Transformer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param attention_probs_dropout_prob: dropout probability for attention layers
    :type attention_probs_dropout_prob: bool
    :param hidden_dropout_prob: dropout probability for hidden layers
    :type hidden_dropout_prob: bool
    :param depth: the 3D parallelism depth
    :type depth: int
    :param input_parallel_mode: parallel mode of input tensor
    :type input_parallel_mode: ParallelMode
    :param weight_parallel_mode: parallel mode of weight
    :type weight_parallel_mode: ParallelMode
    :param dtype: dtype of parameters, defaults to None
    :type dtype: dtype, optional
    :param bias: whether to add bias, defaults to True
    :type bias: bool, optional
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_probs_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()
        self.depth = get_depth_from_env()
        # self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        # self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        # self.output_parallel_mode = get_last_group(self.input_parallel_mode,
        #                                            self.weight_parallel_mode)
        self.hidden_size = hidden_size
        self.num_attention_heads = divide(num_attention_heads, self.depth)
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.checkpoint = checkpoint
        self.init_weight = 'torch'
        self.init_bias = 'torch'
        if init_method == 'jax':
            self.init_weight = 'jax'
            self.init_bias = 'zero'

        self.query_key_value = Linear3D(
            self.hidden_size,
            3 * self.hidden_size,
            # self.input_parallel_mode,
            # self.weight_parallel_mode,
            dtype=dtype,
            bias=bias,
            init_weight=self.init_weight,
            init_bias=self.init_bias)
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.dense = Linear3D(
            self.hidden_size,
            self.hidden_size,
            #   self.output_parallel_mode,
            #   self.weight_parallel_mode,
            dtype=dtype,
            bias=bias,
            init_weight=self.init_weight,
            init_bias=self.init_bias)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

    # def groups_for_next_layer(self) -> Tuple[ParallelMode, ParallelMode]:
    #     return self.input_parallel_mode, self.weight_parallel_mode

    def _forward(self, hidden_states: Tensor) -> Tensor:
        query_key_value = self.query_key_value(hidden_states)
        new_qkv_shape = query_key_value.shape[:-1] + \
            (self.num_attention_heads, 3 * self.attention_head_size)
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))
        query_layer, key_layer, value_layer = torch.chunk(query_key_value,
                                                          3,
                                                          dim=-1)

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.dense(context_layer)
        with seed(ParallelMode.TENSOR):
            output = self.dropout(output)

        return output

    def _checkpoint_forward(self, hidden_states: Tensor) -> Tensor:
        return checkpoint(self._forward, hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states)
        else:
            return self._forward(hidden_states)


@LAYERS.register_module
class ViTMLP3D(nn.Module):
    """[summary]

    :param hidden_size: hidden size
    :type hidden_size: int
    :param mlp_ratio: hidden size of MLP divided by embedding dim
    :type mlp_ratio: int
    :param hidden_dropout_prob: dropout probability for hidden layers
    :type hidden_dropout_prob: float
    :param hidden_act: activation function for hidden layers
    :type hidden_act: str
    :param depth: the 3D parallelism depth
    :type depth: int
    :param input_parallel_mode: parallel mode of input tensor
    :type input_parallel_mode: ParallelMode
    :param weight_parallel_mode: parallel mode of weight
    :type weight_parallel_mode: ParallelMode
    :param dtype: dtype of parameters, defaults to None
    :type dtype: dtype, optional
    :param bias: whether to add bias, defaults to True
    :type bias: bool, optional
    """

    def __init__(self,
                 hidden_size: int,
                 mlp_ratio: int,
                 hidden_dropout_prob: float,
                 hidden_act: str = 'gelu',
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()
        # self.depth = get_depth_from_env()
        # self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        # self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        # self.output_parallel_mode = get_last_group(self.input_parallel_mode,
        #                                            self.weight_parallel_mode)
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.checkpoint = checkpoint
        self.init_weight = init_method
        self.init_bias = init_method

        self.dense_1 = Linear3D(
            self.hidden_size,
            self.mlp_ratio * self.hidden_size,
            # self.input_parallel_mode,
            # self.weight_parallel_mode,
            dtype=dtype,
            bias=bias,
            init_weight=self.init_weight,
            init_bias=self.init_bias)
        self.activation_func = ACT2FN[hidden_act]
        self.dense_2 = Linear3D(
            self.mlp_ratio * self.hidden_size,
            self.hidden_size,
            # self.output_parallel_mode,
            # self.weight_parallel_mode,
            dtype=dtype,
            bias=bias,
            init_weight=self.init_weight,
            init_bias=self.init_bias)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    # def groups_for_next_layer(self) -> Tuple[ParallelMode, ParallelMode]:
    #     return self.input_parallel_mode, self.weight_parallel_mode

    def _forward(self, hidden_states: Tensor) -> Tensor:
        intermediate_output = self.dense_1(hidden_states)
        intermediate_output = self.activation_func(intermediate_output)
        with seed(ParallelMode.TENSOR):
            intermediate_output = self.dropout(intermediate_output)
        output = self.dense_2(intermediate_output)
        with seed(ParallelMode.TENSOR):
            output = self.dropout(output)
        return output

    def _checkpoint_forward(self, hidden_states: Tensor) -> Tensor:
        return checkpoint(self._forward, hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states)
        else:
            return self._forward(hidden_states)


@LAYERS.register_module
class ViTHead3D(nn.Module):
    """Output layer for 3D parallel Vision Transformer

    :param in_features: size of input tensor
    :type in_features: int
    :param num_classes: number of classes
    :type num_classes: int
    :param depth: the 3D parallelism depth
    :type depth: int
    :param input_parallel_mode: parallel mode of input tensor
    :type input_parallel_mode: ParallelMode
    :param weight_parallel_mode: parallel mode of weight
    :type weight_parallel_mode: ParallelMode
    :param dtype: dtype of parameters, defaults to None
    :type dtype: dtype, optional
    :param bias: whether to add bias, defaults to True
    :type bias: bool, optional
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        # self.depth = get_depth_from_env()
        # self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        # self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        # self.output_parallel_mode = get_last_group(self.input_parallel_mode,
        #                                            self.weight_parallel_mode)
        self.in_features = in_features
        self.num_classes = num_classes
        # out_features = math.ceil(self.num_classes /
        #                          (self.depth**2)) * (self.depth**2)
        # self.num_classes_per_partition = divide(self.num_classes, self.depth)
        self.init_weight = 'torch'
        self.init_bias = 'torch'
        if init_method == 'jax':
            self.init_weight = 'zero'
            self.init_bias = 'zero'

        self.linear = Classifier3D(
            self.in_features,
            self.num_classes,
            #    self.input_parallel_mode,
            #    self.weight_parallel_mode,
            dtype=dtype,
            bias=bias,
            init_weight=self.init_weight,
            init_bias=self.init_bias)

    def forward(self, x: Tensor) -> Tensor:
        # [b/q^2, s, h/q] --> [b/q^2, h/q]
        x = x[:, 0]
        # [b/q^2, h/q] --> [b/q^2, c/q]
        x = self.linear(x)
        # return x[:, :self.num_classes_per_partition]
        return x

    def extra_repr(self):
        return 'in_features={}, num_classes={}'.format(self.in_features,
                                                       self.num_classes)
