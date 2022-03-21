import math
from typing import Callable

import torch
from colossalai import nn as col_nn
from colossalai.builder.pipeline import partition_uniform
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.utils import CheckpointModule, divide
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.registry import LAYERS, LOSSES, MODELS
from colossalai.utils import get_current_device
from torch import dtype, nn

__all__ = [
    'GPT', 'GPTLMLoss', 'gpt2_small', 'gpt2_medium', 'gpt2_large', 'gpt2_xl', 'gpt2_8B', 'gpt2_xl_pipeline',
    'gpt2_8B_pipeline', 'gpt3', 'gpt3_pipeline'
]


@LAYERS.register_module
class GPTEmbedding(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 max_position_embeddings: int,
                 num_tokentypes: int = 0,
                 padding_idx: int = None,
                 dropout: float = 0.,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.word_embeddings = col_nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, dtype=dtype)
        self.position_embeddings = col_nn.Embedding(max_position_embeddings, embedding_dim, dtype=dtype)
        if num_tokentypes > 0:
            self.tokentype_embeddings = col_nn.Embedding(num_tokentypes, embedding_dim, dtype=dtype)
        else:
            self.tokentype_embeddings = None
        self.dropout = col_nn.Dropout(dropout)

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=get_current_device()).unsqueeze(0)
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        if self.tokentype_embeddings is not None and tokentype_ids is not None:
            x = x + self.tokentype_embeddings(tokentype_ids)
        x = self.dropout(x)

        return x


@LAYERS.register_module
class GPTSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 fuse_scale_mask_softmax: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.fuse_scale_mask_softmax = fuse_scale_mask_softmax
        self.attention_head_size = divide(dim, num_heads)
        self.query_key_value = col_nn.Linear(dim, 3 * dim, dtype=dtype, bias=bias)
        if fuse_scale_mask_softmax:
            from colossalai.kernel import FusedScaleMaskSoftmax
            from colossalai.kernel.cuda_native.scaled_softmax import \
                AttnMaskType
            self.softmax = FusedScaleMaskSoftmax(input_in_fp16=True,
                                                 input_in_bf16=False,
                                                 attn_mask_type=AttnMaskType.causal,
                                                 scaled_masked_softmax_fusion=True,
                                                 mask_func=None,
                                                 softmax_in_fp32=True,
                                                 scale=math.sqrt(self.attention_head_size))
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = col_nn.Dropout(attention_dropout)
        self.dense = col_nn.Linear(dim, dim, dtype=dtype, bias=True)
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = divide(all_head_size, self.attention_head_size)
        new_qkv_shape = qkv.shape[:-1] + \
            (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))

        if self.fuse_scale_mask_softmax:
            x = self.softmax(x, attention_mask)
        else:
            x = x / math.sqrt(self.attention_head_size)
            # causal mask
            q_len, k_len = q.size(-2), k.size(-2)
            causal_mask = torch.tril(torch.ones((q_len, k_len), dtype=torch.uint8,
                                                device=get_current_device())).view(1, 1, q_len, k_len).bool()
            x = torch.where(causal_mask, x, torch.tensor(-1e4, dtype=x.dtype, device=get_current_device()))
            if attention_mask is not None:
                x = x + attention_mask
            x = self.softmax(x)

        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x


@LAYERS.register_module
class GPTMLP(nn.Module):

    def __init__(self,
                 dim: int,
                 mlp_ratio: float,
                 activation: Callable,
                 dropout: float,
                 dtype: dtype = None,
                 bias: bool = True):
        super().__init__()
        intermediate_dim = int(dim * mlp_ratio)
        self.dense_1 = col_nn.Linear(dim, intermediate_dim, dtype=dtype, bias=bias)
        self.activation = activation
        self.dense_2 = col_nn.Linear(intermediate_dim, dim, dtype=dtype, bias=bias)
        self.dropout = col_nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


@LAYERS.register_module
class GPTBlock(CheckpointModule):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 layernorm_epsilon: float = 1e-5,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False):
        super().__init__(checkpoint)
        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.attn = GPTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                                     dtype=dtype)
        self.norm2 = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = GPTMLP(dim=dim, mlp_ratio=mlp_ratio, activation=activation, dropout=dropout, dtype=dtype, bias=bias)

    def _forward(self, x, attention_mask=None):
        if not self.apply_post_layernorm:
            residual = x
        x = self.norm1(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.attn(x, attention_mask)

        if not self.apply_post_layernorm:
            residual = x
        x = self.norm2(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.mlp(x)

        return x, attention_mask


@LAYERS.register_module
class GPTLMHead(nn.Module):

    def __init__(self,
                 dim: int,
                 vocab_size: int,
                 word_embeeding_weight: nn.Parameter = None,
                 bias: bool = False,
                 dtype: dtype = None) -> None:
        super().__init__()
        self.dense = col_nn.Classifier(dim, vocab_size, word_embeeding_weight, bias=bias, dtype=dtype)

    @property
    def weight(self):
        return self.dense.weight

    def forward(self, x):
        x = self.dense(x)
        return x


@LOSSES.register_module
class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = col_nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


@MODELS.register_module
class GPT(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False) -> None:
        super().__init__()
        self.embed = GPTEmbedding(embedding_dim=dim,
                                  vocab_size=vocab_size,
                                  max_position_embeddings=max_position_embeddings,
                                  padding_idx=padding_idx,
                                  dropout=embedding_dropout,
                                  dtype=dtype)
        self.blocks = nn.ModuleList([
            GPTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                attention_dropout=attention_dropout,
                dropout=dropout,
                layernorm_epsilon=layernorm_epsilon,
                dtype=dtype,
                bias=bias,
                apply_post_layernorm=apply_post_layernorm,
                fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                checkpoint=checkpoint,
            ) for _ in range(depth)
        ])

        self.norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        self.head = GPTLMHead(dim=dim,
                              vocab_size=vocab_size,
                              word_embeeding_weight=self.embed.word_embedding_weight,
                              dtype=dtype)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        if attention_mask is not None:
            batch_size = input_ids.shape[0]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = col_nn.partition_batch(attention_mask)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)    # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        x = self.head(self.norm(x))

        return x


class PipelineGPT(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 max_position_embeddings: int = 1024,
                 dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 layernorm_epsilon: float = 1e-5,
                 activation: Callable = nn.functional.gelu,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 apply_post_layernorm: bool = False,
                 fuse_scale_mask_softmax: bool = False,
                 checkpoint: bool = False,
                 first: bool = False,
                 last: bool = False):
        super().__init__()
        self.checkpoint = checkpoint
        self.first = first
        self.last = last
        if first:
            self.embed = GPTEmbedding(embedding_dim=dim,
                                      vocab_size=vocab_size,
                                      max_position_embeddings=max_position_embeddings,
                                      padding_idx=padding_idx,
                                      dropout=embedding_dropout,
                                      dtype=dtype)
        self.blocks = nn.ModuleList([
            GPTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                attention_dropout=attention_dropout,
                dropout=dropout,
                layernorm_epsilon=layernorm_epsilon,
                dtype=dtype,
                bias=bias,
                apply_post_layernorm=apply_post_layernorm,
                fuse_scale_mask_softmax=fuse_scale_mask_softmax,
                checkpoint=checkpoint,
            ) for _ in range(depth)
        ])
        if self.last:
            self.norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)
            self.head = GPTLMHead(dim=dim, vocab_size=vocab_size, dtype=dtype)

    def forward(self, x=None, input_ids=None, attention_mask=None):
        if self.first:
            x = self.embed(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        if attention_mask is not None:
            if self.first:
                batch_size = input_ids.shape[0]
            else:
                batch_size = x.shape[0]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = col_nn.partition_batch(attention_mask)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)    # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        if self.last:
            x = self.head(self.norm(x))

        return x


def _create_gpt_model(**model_kwargs):
    model = GPT(**model_kwargs)
    return model


def _create_gpt_pipeline_model(depth=48, num_chunks=1, layer_partitions=None, **model_kwargs):
    logger = get_dist_logger()
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    rank = gpc.get_global_rank()
    wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
    parts = partition_uniform(depth, pipeline_size,
                              num_chunks)[pipeline_rank] if layer_partitions is None else layer_partitions
    models = []
    for start, end in parts:
        model_kwargs['first'] = start == 0
        model_kwargs['last'] = end == depth
        model_kwargs['depth'] = end - start
        chunk = PipelineGPT(**model_kwargs).to(get_current_device())
        if start == 0:
            wrapper.register_parameter(chunk.embed.word_embedding_weight)
        elif end == depth:
            wrapper.register_parameter(chunk.head.weight)
        models.append(chunk)
        logger.info(f'==> Rank {rank} built layer {start}-{end} / total {depth}')
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model


@MODELS.register_module
def gpt2_small(**kwargs):
    model_kwargs = dict(dim=768, depth=12, num_heads=12, **kwargs)
    return _create_gpt_model(**model_kwargs)


@MODELS.register_module
def gpt2_medium(**kwargs):
    model_kwargs = dict(dim=1024, depth=24, num_heads=8, **kwargs)
    return _create_gpt_model(**model_kwargs)


@MODELS.register_module
def gpt2_large(**kwargs):
    model_kwargs = dict(dim=1536, depth=36, num_heads=12, **kwargs)
    return _create_gpt_model(**model_kwargs)


@MODELS.register_module
def gpt2_xl(**kwargs):
    model_kwargs = dict(dim=1600, depth=48, num_heads=16, **kwargs)
    return _create_gpt_model(**model_kwargs)


@MODELS.register_module
def gpt2_8B(**kwargs):
    model_kwargs = dict(dim=3072, depth=72, num_heads=24, **kwargs)
    return _create_gpt_model(**model_kwargs)


@MODELS.register_module
def gpt2_xl_pipeline(**kwargs):
    model_kwargs = dict(dim=1600, depth=48, num_heads=20, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)


@MODELS.register_module
def gpt2_8B_pipeline(**kwargs):
    model_kwargs = dict(dim=3072, depth=72, num_heads=24, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)


@MODELS.register_module
def gpt3(**kwargs):
    model_kwargs = dict(dim=12288, depth=96, num_heads=96, **kwargs)
    return _create_gpt_model(**model_kwargs)


@MODELS.register_module
def gpt3_pipeline(**kwargs):
    model_kwargs = dict(dim=12288, depth=96, num_heads=96, **kwargs)
    return _create_gpt_pipeline_model(**model_kwargs)
