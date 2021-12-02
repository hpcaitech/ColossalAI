#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from colossalai import context

import torch
from torch import nn as nn, Tensor, distributed as dist

from colossalai.context import seed, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer._common_utils import divide, ACT2FN
from colossalai.nn.layer.vanilla_vision_transformer.layers import to_2tuple
from colossalai.registry import LAYERS
from colossalai.utils import checkpoint
from colossalai.utils import get_current_device
from .layers import Linear1D_Col, Linear1D_Row
from .._common_utils import set_tensor_parallel_attribute
from ..base_layer import ParallelLayer


@LAYERS.register_module
class ViTMLP1D(ParallelLayer):
    """MLP layer for 1D parallel Vision Transformer

    :param in_features: size of each input sample
    :type in_features: int
    :param mlp_ratio: hidden size of MLP divided by embedding dim
    :type mlp_ratio: int
    :param act_func: activation function, defaults to 'gelu'
    :type act_func: str, optional
    :param dropout_prob: dropout probability, defaults to 0.
    :type dropout_prob: float, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param checkpoint: whether to checkpoint the layer, defaults to False
    :type checkpoint: bool, optional
    """

    def __init__(self,
                 in_features: int,
                 mlp_ratio: int,
                 act_func: str = 'gelu',
                 dropout_prob: float = 0.,
                 dtype=None,
                 checkpoint: bool = False,
                 skip_bias_add: bool = False
                 ):
        super().__init__()

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.checkpoint = checkpoint
        self.skip_bias_add = skip_bias_add
        self.bias = not skip_bias_add
        # Project to mlp_ratio * h.
        self.dense_1 = Linear1D_Col(
            self.in_features,
            int(self.mlp_ratio * self.in_features),
            bias=not skip_bias_add,
            dtype=dtype,
            gather_output = False,
        )

        self.act = ACT2FN[act_func]

        # Project back to h.
        self.dense_2 = Linear1D_Row(
            int(self.mlp_ratio * self.in_features),
            self.in_features,
            bias=not skip_bias_add,
            dtype=dtype,
            parallel_input = True,
        )

        self.dropout = nn.Dropout(dropout_prob)

    def _forward(self, hidden_states: Tensor) -> Tensor:
        intermediate_output = self.dense_1(hidden_states)
        intermediate_output = self.act(intermediate_output)

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
class ViTSelfAttention1D(ParallelLayer):
    """Self-attention layer for 1D parallel Vision Transformer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param attention_dropout_prob: dropout probability for attention layers
    :type attention_dropout_prob: float
    :param hidden_dropout_prob: dropout probability for hidden layers
    :type hidden_dropout_prob: float
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param checkpoint: whether to checkpoint the layer, defaults to False
    :type checkpoint: bool, optional
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 checkpoint: bool = False
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)

        self.checkpoint = checkpoint

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
        self.softmax = nn.Softmax(dim=-1)

    def _forward(self, hidden_states: Tensor) -> Tensor:
        query_key_value = self.query_key_value(hidden_states)
        new_qkv_shape = query_key_value.shape[:-1] + \
                        (self.num_attention_heads_per_partition, 3 * self.attention_head_size)
        query_key_value = query_key_value.view(new_qkv_shape)
        query_key_value = query_key_value.permute((0, 2, 1, 3))
        query_layer, key_layer, value_layer = torch.chunk(
            query_key_value, 3, dim=-1)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
                           math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)

        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.hidden_size_per_partition,)
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
class ViTSelfAttention1DV2(ParallelLayer):
    """Self-attention layer for 1D parallel Vision Transformer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_attention_heads: number of attention heads
    :type num_attention_heads: int
    :param attention_dropout_prob: dropout probability for attention layers
    :type attention_dropout_prob: float
    :param hidden_dropout_prob: dropout probability for hidden layers
    :type hidden_dropout_prob: float
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param checkpoint: whether to checkpoint the layer, defaults to False
    :type checkpoint: bool, optional
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 checkpoint: bool = False
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)

        self.checkpoint = checkpoint

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
        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.num_attention_heads ** -0.5

    def _forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.query_key_value(x).reshape(B, N, 3, self.num_attention_heads, C //
                                  (self.num_attention_heads * gpc.tensor_parallel_size)).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        with seed(ParallelMode.TENSOR):
            attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C // gpc.tensor_parallel_size)
        x = self.dense(x)
        with seed(ParallelMode.TENSOR):
            x = self.dropout(x)
        return x

    def _checkpoint_forward(self, hidden_states: Tensor) -> Tensor:
        return checkpoint(self._forward, hidden_states)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states)
        else:
            return self._forward(hidden_states)

@LAYERS.register_module
class ViTHead1D(ParallelLayer):
    """Output layer for 1D parallel Vision Transformer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_classes: number of classes
    :type num_classes: int
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 hidden_size,
                 num_classes,
                 dtype=None,
                 ):
        super().__init__()

        self.linear = Linear1D_Col(
            hidden_size,
            num_classes,
            dtype=dtype,
            gather_output = True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, 0]
        x = self.linear(x)
        return x

@LAYERS.register_module
class ViTHeadNormal(ParallelLayer):
    """Output layer for 1D parallel Vision Transformer

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_classes: number of classes
    :type num_classes: int
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 hidden_size,
                 num_classes,
                 dtype=None,
                 ):
        super().__init__()

        self.linear = nn.Linear(
            hidden_size,
            num_classes,
            dtype = dtype
        )
        self._broadcast_linear_params()
    
    def _broadcast_linear_params(self) -> None:
        self.to(get_current_device())
        ranks = gpc.get_ranks_in_group(ParallelMode.PARALLEL_1D)

        dist.broadcast(self.linear.weight, src=ranks[0],
                       group=gpc.get_group(ParallelMode.PARALLEL_1D))
        dist.broadcast(self.linear.bias, src=ranks[0],
                       group=gpc.get_group(ParallelMode.PARALLEL_1D))

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, 0]
        x = self.linear(x)
        return x

@LAYERS.register_module
class ViTPatchEmbedding1D(ParallelLayer):
    """ 2D Image to Patch Embedding

    :param img_size: iamge size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param embed_dim: dimension of embedding
    :type embed_dim: int
    :param in_chans: number of channels of input image, defaults to 3
    :type in_chans: int, optional
    :param flatten: whether to flatten output tensor, defaults to True
    :type flatten: bool, optional
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 embed_dim,
                 in_chans=3,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_dim = embed_dim

        with seed(ParallelMode.TENSOR):
            self.proj = nn.Conv2d(in_chans,
                                  self.embed_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size
                                  )

        # sync
        self._broadcast_conv_params()
        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute(self.proj.weight)
        set_tensor_parallel_attribute(self.proj.bias)

    def _broadcast_conv_params(self) -> None:
        self.to(get_current_device())
        ranks = gpc.get_ranks_in_group(ParallelMode.PARALLEL_1D)

        dist.broadcast(self.proj.weight, src=ranks[0],
                       group=gpc.get_group(ParallelMode.PARALLEL_1D))
        dist.broadcast(self.proj.bias, src=ranks[0],
                       group=gpc.get_group(ParallelMode.PARALLEL_1D))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


@LAYERS.register_module
class ViTTokenFuser1D(ParallelLayer):
    """
    Fuse cls token and pos embedding to the input

    :param img_size: image size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param embed_dim: dimension of embedding
    :type embed_dim: int
    :param drop_rate: dropout probability, defaults to 0.
    :type drop_rate: float, optional
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 embed_dim,
                 drop_rate=0.
                 ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.num_patches + 1, self.embed_dim))

        # move to cuda before broadcast
        self.to(get_current_device())

        # sync param in both forward and backward
        _cls_token = self.cls_token.view(-1)
        _pos_embed = self.pos_embed.view(-1)
        self._param = torch.cat([_cls_token, _pos_embed], dim=0)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute(self.cls_token)
        set_tensor_parallel_attribute(self.pos_embed)

    def forward(self, x: Tensor) -> Tensor:
        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        with seed(ParallelMode.TENSOR):
            x = self.pos_drop(x + self.pos_embed)
        return x.contiguous()

