import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomModel

import colossalai.shardformer.layer as col_nn

from .._utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class BloomPolicy(Policy):

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        # TODO:
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        return {
            BloomBlock:
                ModulePolicyDescription(
                    attribute_replacement={
        # 1. shard hidden size
                        "self_attention.hidden_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                        "self_attention.split_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
        # 2. shard number of heads
                        "self_attention.num_heads":
                            self.model.config.n_head // self.shard_config.tensor_parallel_size,
                    },
                    param_replacement=[],
                    sub_module_replacement=[
        # SubModuleReplacementDescription(
        #     suffix="input_layernorm",
        #     target_module=col_nn.LayerNorm,
        #     kwargs={"use_mixedfusedLN": self.shard_config.use_mixedfusedLN},
        # ),
                        SubModuleReplacementDescription(suffix="self_attention.query_key_value",
                                                        target_module=col_nn.LinearFused1D_Col,
                                                        kwargs={'n_fused': 3}),
                        SubModuleReplacementDescription(
                            suffix="self_attention.dense",
                            target_module=col_nn.Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="self_attention.attention_dropout",
                            target_module=col_nn.Dropout1D,
                        ),
        # SubModuleReplacementDescription(
        #     suffix="post_attention_layernorm",
        #     target_module=col_nn.LayerNorm,
        #     kwargs={"use_mixedfusedLN": self.shard_config.use_mixedfusedLN},
        # ),
                        SubModuleReplacementDescription(
                            suffix="mlp.dense_h_to_4h",
                            target_module=col_nn.Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="mlp.dense_4h_to_h",
                            target_module=col_nn.Linear1D_Row,
                        ),
                    ]),
            BloomModel:
                ModulePolicyDescription(
                    attribute_replacement={
                        "num_heads": self.model.config.n_head // self.shard_config.tensor_parallel_size,
                    },
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="word_embeddings",
                            target_module=col_nn.VocabParallelEmbedding1D,
                        ),
        # SubModuleReplacementDescription(
        #     suffix="word_embeddings_layernorm",
        #     target_module=col_nn.LayerNorm,
        #     kwargs={"use_mixedfusedLN": self.shard_config.use_mixedfusedLN},
        # ),
        # SubModuleReplacementDescription(
        #     suffix="ln_f",
        #     target_module=col_nn.LayerNorm,
        #     kwargs={"use_mixedfusedLN": self.shard_config.use_mixedfusedLN},
        # ),
                    ])
        }

    def new_model_class(self):
        # do nothing
        return self.model

    def postprocess(self):
        return self.model


# BertModel
class BloomModelPolicy(BloomPolicy):

    def __init__(self) -> None:
        super().__init__()
