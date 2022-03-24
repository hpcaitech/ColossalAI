from typing import Callable, List
from torch import dtype, nn
from colossalai import nn as col_nn
from colossalai.registry import LAYERS, MODELS
from colossalai.nn.layer import MoeModule
from colossalai.context import MOE_CONTEXT
from colossalai.logging import get_dist_logger
from colossalai.nn.layer.utils import CheckpointModule, divide
from model_zoo.gpt.gpt import GPTEmbedding, GPTSelfAttention, GPTMLP, GPTBlock, GPTLMHead


@LAYERS.register_module
class MOEGPTBlock(CheckpointModule):

    def __init__(self,
                 num_experts: int,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 activation: Callable,
                 capacity_factor_train: float = 1.0,
                 capacity_factor_eval: float = 1.0,
                 use_residual: bool = False,
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

        mpl_factory_dict = dict(dim=dim,
                                mlp_ratio=mlp_ratio,
                                activation=activation,
                                dropout=dropout,
                                dtype=dtype,
                                bias=bias)

        self.mlp = MoeModule(dim_model=dim,
                             num_experts=num_experts,
                             top_k=1,
                             capacity_factor_train=capacity_factor_train,
                             capacity_factor_eval=capacity_factor_eval,
                             noisy_policy='Jitter',
                             use_residual=use_residual,
                             expert_cls=GPTMLP,
                             **mpl_factory_dict)

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


@MODELS.register_module
class MOEGPT(nn.Module):

    def __init__(self,
                 num_experts: int or List[int],
                 use_residual: bool = False,
                 capacity_factor_train: float = 1.0,
                 capacity_factor_eval: float = 1.0,
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

        half_depth = divide(depth, 2)
        if isinstance(num_experts, list):
            assert len(num_experts) == half_depth, \
                "The length of num_experts should equal to the number of MOE layers"
            num_experts_list = num_experts
        else:
            num_experts_list = [num_experts] * half_depth

        self.embed = GPTEmbedding(embedding_dim=dim,
                                  vocab_size=vocab_size,
                                  max_position_embeddings=max_position_embeddings,
                                  padding_idx=padding_idx,
                                  dropout=embedding_dropout,
                                  dtype=dtype)

        block_list = []
        block_factory_dict = dict(dim=dim,
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
                                  checkpoint=checkpoint)

        for i in range(depth):

            if i % 2 == 0:
                block_module = GPTBlock(**block_factory_dict)
            else:
                num_experts = num_experts_list[i // 2]
                block_module = MOEGPTBlock(num_experts=num_experts,
                                           capacity_factor_train=capacity_factor_train,
                                           capacity_factor_eval=capacity_factor_eval,
                                           use_residual=use_residual,
                                           **block_factory_dict)

            block_list.append(block_module)

        self.blocks = nn.ModuleList(block_list)

        self.norm = col_nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon, dtype=dtype)

        self.head = GPTLMHead(dim=dim,
                              vocab_size=vocab_size,
                              word_embeeding_weight=self.embed.word_embedding_weight,
                              dtype=dtype)

    def forward(self, input_ids, attention_mask=None):
        MOE_CONTEXT.reset_loss()
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


def _create_moegpt_model(**model_kwargs):
    model = MOEGPT(**model_kwargs)
    return model


def _prmoe_check_sanity(kwargs_dict):
    logger = get_dist_logger()
    if not kwargs_dict.pop('use_residual', False):
        logger.warning(
            "If you want to use PR-MOE, please set 'use_residual' to True. "
            "Otherwise, we'll force 'use_residual' to True.",
            ranks=[0])


@MODELS.register_module
def prmoe_4b(**kwargs):
    _prmoe_check_sanity(kwargs)
    model_kwargs = dict(num_experts=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64],
                        use_residual=True,
                        dim=1024,
                        depth=24,
                        num_heads=16,
                        **kwargs)
    return _create_moegpt_model(**model_kwargs)


@MODELS.register_module
def prmoe_31b(**kwargs):
    _prmoe_check_sanity(kwargs)
    model_kwargs = dict(num_experts=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 128, 128],
                        use_residual=True,
                        dim=2048,
                        depth=24,
                        num_heads=16,
                        **kwargs)
    return _create_moegpt_model(**model_kwargs)


@MODELS.register_module
def prmoe_51b(**kwargs):
    _prmoe_check_sanity(kwargs)
    model_kwargs = dict(num_experts=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64],
                        use_residual=True,
                        dim=3072,
                        depth=32,
                        num_heads=24,
                        **kwargs)
    return _create_moegpt_model(**model_kwargs)
