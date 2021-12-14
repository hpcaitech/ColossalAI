#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from colossalai.utils.activation_checkpoint import checkpoint
import torch
import torch
from torch import nn as nn, Tensor, distributed as dist
from torch.nn import functional as F

from colossalai.context import seed, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer._common_utils import divide, ACT2FN
from colossalai.registry import LAYERS, MODELS
from colossalai.utils import checkpoint
from colossalai.utils import get_current_device
from colossalai.nn.layer import Linear1D_Col, Linear1D_Row
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.nn.layer.fused_bias_gelu import bias_gelu_impl

__all__ = [
    'GPT1D',
    'GPT2_small_1D',
    'GPT2_medium_1D',
    'GPT2_large_1D',
    'GPT2_exlarge_1D',
    'GPT3_1D',
]


@MODELS.register_module
class GPT1D(nn.Module):
    def __init__(self,
                 vocab_size: int = 50256,
                 depth: int = 12,
                 num_heads: int = 12,
                 embed_dim: int = 768,
                 mlp_ratio: int = 4.0,
                 max_position_embeddings: int = 1024,
                 drop_rate: float = 0.,
                 embed_drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 act_func: str = 'gelu',
                 checkpoint: bool = False,
                 dtype=None):
        super().__init__()

        self.embed = GPTEmbedding1D(embed_dim,
                                    vocab_size,
                                    max_position_embeddings,
                                    embed_drop_rate,
                                    weight_init='torch')

        self.blocks = nn.ModuleList([
            GPTTransformerLayer1D(embed_dim, num_heads, act_func, mlp_ratio, attn_drop_rate,
                                  drop_rate, dtype, checkpoint, max_position_embeddings)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = GPTLMHead1D()
        self.dtype = dtype

    def forward(self, input_ids, attention_mask):
        # input_ids, attention_mask = x
        hidden_states = self.embed(input_ids=input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        batch_size = hidden_states.shape[0]
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            hidden_states, attention_mask = block(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        lm_logits = self.head(hidden_states, self.embed.wte.weight)
        return lm_logits


def _create_gpt_model(**model_kwargs):
    model = GPT1D(**model_kwargs)
    return model


@MODELS.register_module
def GPT2_small_1D(checkpoint=False):
    cfg = dict(embed_dim=768, depth=12, num_heads=12, checkpoint=checkpoint)
    return _create_gpt_model(**cfg)


@MODELS.register_module
def GPT2_medium_1D(checkpoint=False):
    cfg = dict(embed_dim=1024, depth=24, num_heads=16, checkpoint=checkpoint)
    return _create_gpt_model(**cfg)


@MODELS.register_module
def GPT2_large_1D(checkpoint=False):
    cfg = dict(embed_dim=1280, depth=36, num_heads=20, checkpoint=checkpoint)
    return _create_gpt_model(**cfg)


@MODELS.register_module
def GPT2_exlarge_1D(checkpoint=False):
    cfg = dict(embed_dim=1600, depth=48, num_heads=25, checkpoint=checkpoint)
    return _create_gpt_model(**cfg)


@MODELS.register_module
def GPT3_1D(checkpoint=False):
    cfg = dict(embed_dim=12288, depth=96, num_heads=96, checkpoint=checkpoint)
    return _create_gpt_model(**cfg)


@LAYERS.register_module
class GPTMLP1D(ParallelLayer):
    """MLP layer for 1D parallel GPT

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
                 skip_bias_add: bool = False,
                 weight_init='torch'
                 ):
        super().__init__()

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.checkpoint = checkpoint
        self.skip_bias_add = skip_bias_add
        assert weight_init in ('torch', 'jax')

        if act_func == 'fused_gelu':
            self.act = bias_gelu_impl
            skip_dense_1_add_bias = True
        else:
            self.act = ACT2FN[act_func]
            skip_dense_1_add_bias = False

        # Project to mlp_ratio * h.
        self.dense_1 = Linear1D_Col(
            self.in_features,
            int(self.mlp_ratio * self.in_features),
            dtype=dtype,
            gather_output=False,
            skip_bias_add=skip_dense_1_add_bias,
            init_weight=weight_init,
            init_bias=weight_init
        )

        # Project back to h.
        self.dense_2 = Linear1D_Row(
            int(self.mlp_ratio * self.in_features),
            self.in_features,
            dtype=dtype,
            parallel_input=True,
            init_weight=weight_init, init_bias=weight_init
        )

        self.dropout = nn.Dropout(dropout_prob)

    def _forward(self, hidden_states: Tensor) -> Tensor:
        if self.act == bias_gelu_impl:
            intermediate_output, bias = self.dense_1(hidden_states)
            intermediate_output = self.act(intermediate_output, bias)
        else:
            intermediate_output = self.dense_1(hidden_states)
            intermediate_output = self.act(intermediate_output)

        with seed(ParallelMode.TENSOR):
            intermediate_output = self.dropout(intermediate_output)
        output = self.dense_2(intermediate_output)
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
class GPTSelfAttention1D(ParallelLayer):
    """Self-attention layer for 1D parallel GPT
    """

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout_prob: float,
                 hidden_dropout_prob: float,
                 dtype=None,
                 checkpoint: bool = False,
                 weight_init='torch',
                 max_position_embeddings=1024,
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_head_size = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, gpc.tensor_parallel_size)
        self.hidden_size_per_partition = divide(hidden_size, gpc.tensor_parallel_size)

        self.checkpoint = checkpoint
        assert weight_init in ('torch', 'jax')
        if weight_init == 'jax':
            init_bias = 'zero'
        else:
            init_bias = weight_init

        self.query_key_value = Linear1D_Col(
            hidden_size,
            3 * hidden_size,
            dtype=dtype,
            init_weight=weight_init,
            init_bias=init_bias
        )
        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.dense = Linear1D_Row(
            hidden_size,
            hidden_size,
            dtype=dtype,
            parallel_input=True,
            init_weight=weight_init, init_bias=init_bias
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        max_positions = max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def _forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
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

        # causal mask
        query_length, key_length = query_layer.size(-2), key_layer.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attention_scores = torch.where(causal_mask, attention_scores, self.masked_bias.to(attention_scores))

        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attention_scores = attention_scores.type(value_layer.dtype)

        with seed(ParallelMode.TENSOR):
            attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.dense(context_layer)
        output = self.dropout(output)

        return output

    def _checkpoint_forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        return checkpoint(self._forward, hidden_states, attention_mask)

    def forward(self, hidden_states: Tensor, attention_mask=None) -> Tensor:
        if self.checkpoint:
            return self._checkpoint_forward(hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)


@LAYERS.register_module
class GPTTransformerLayer1D(ParallelLayer):
    """Pre-Layernorm Transformer layer which contains a self-attention layer and a MLP layer.

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
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 ):
        super().__init__()

        self.dtype = dtype
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = GPTSelfAttention1D(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            dtype=dtype,
            max_position_embeddings=max_position_embeddings,
            checkpoint=checkpoint,
        )
        #checkpoint: bool = False
        # weight_init='torch'

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = GPTMLP1D(
            in_features=hidden_size,
            dropout_prob=hidden_dropout_prob,
            act_func=act_func,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            checkpoint=checkpoint,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        output = (hidden_states, attention_mask)
        return output


@LAYERS.register_module
class GPTEmbedding1D(ParallelLayer):
    """Word embedding for GPT1D.

    :param hidden_size: hidden size
    :type hidden_size: int
    :param num_classes: number of classes
    :type num_classes: int
    :param dtype: dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 max_position_embeddings: int,
                 dropout_embed: float = 0.,
                 weight_init: str = 'torch'):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout_embed = nn.Dropout(dropout_embed)

        self.wte = nn.Embedding(vocab_size, self.embed_dim, padding_idx=50256)
        self.wpe = nn.Embedding(max_position_embeddings, self.embed_dim)

        # sync
        # self._broadcast_emb_params()

    def _broadcast_emb_params(self) -> None:
        self.to(get_current_device())
        ranks = gpc.get_ranks_in_group(ParallelMode.PARALLEL_1D)

        dist.broadcast(self.wte.weight, src=ranks[0],
                       group=gpc.get_group(ParallelMode.PARALLEL_1D))
        dist.broadcast(self.wpe.weight, src=ranks[0],
                       group=gpc.get_group(ParallelMode.PARALLEL_1D))

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids=None, input_embeds=None, position_ids=None, token_type_ids=None) -> Tensor:
        assert input_ids is not None or input_embeds is not None
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif input_embeds is not None:
            input_shape = input_embeds.size()[:-1]

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1] + 0, dtype=torch.long, device=get_current_device())
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if input_embeds is None:
            input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        with seed(ParallelMode.TENSOR):
            hidden_states = self.dropout_embed(hidden_states)
        return hidden_states


@LAYERS.register_module
class GPTLMHead1D(ParallelLayer):
    """
    Language model head that shares the same parameters with the embedding matrix.
    """

    def __init__(self,
                 dtype=None,
                 ):
        super().__init__()

    def forward(self, x: Tensor, word_embeddings_weight) -> Tensor:
        x = F.linear(x, word_embeddings_weight)
        return x


@LAYERS.register_module
class GPTSequenceClassficationHead1D(ParallelLayer):
    """
    a double linear head for NSP task.
    """

    def __init__(self,
                 hidden_dim,
                 num_labels,
                 dtype=None,
                 ):
        super().__init__()
        self.pooler = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, num_labels, bias=False)

    def forward(self, x: Tensor, sequence_index=0) -> Tensor:
        pooled = x[:, sequence_index, :]
        pooled = self.pooler(pooled)
        pooled = torch.tanh(pooled)
        score = self.score(pooled)
        return score
