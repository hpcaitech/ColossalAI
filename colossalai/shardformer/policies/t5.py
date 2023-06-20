import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Stack,
)

from colossalai.shardformer.layer.dropout import Dropout1D
from colossalai.shardformer.layer.layers import Embedding1D, Linear1D_Col, Linear1D_Row

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["T5ModelPolicy", "T5ForConditionalGenerationPolicy", "T5EncoderPolicy"]


class T5ModelPolicy(Policy):

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        return {
            T5Stack:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=Dropout1D,
                                            )
                                        ]),
            T5LayerSelfAttention:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=Dropout1D,
                                            ),
                                        ]),
            T5LayerCrossAttention:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=Dropout1D,
                                            )
                                        ]),
            T5Attention:
                ModulePolicyDescription(attribute_replacement={
                    "d_model":
                        self.model.config.d_model // self.shard_config.tensor_parallel_size,
                    "n_heads":
                        self.model.config.num_heads // self.shard_config.tensor_parallel_size,
                    "inner_dim":
                        self.model.config.num_heads * self.model.config.d_kv // self.shard_config.tensor_parallel_size
                },
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="q",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="k",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="v",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="o",
                                                target_module=Linear1D_Row,
                                            ),
                                            SubModuleReplacementDescription(suffix="relative_attention_bias",
                                                                            target_module=Embedding1D,
                                                                            kwargs=dict(gather_output=False),
                                                                            ignore_if_not_exist=True)
                                        ]),
            T5LayerFF:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=Dropout1D,
                                            ),
                                        ]),
            T5DenseGatedActDense:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="wi_0",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="wi_1",
                                                target_module=Linear1D_Row,
                                            ),
                                            SubModuleReplacementDescription(suffix="wo",
                                                                            target_module=Linear1D_Col,
                                                                            kwargs=dict(gather_output=True)),
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=Dropout1D,
                                            )
                                        ]),
            T5DenseActDense:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="wi",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="wo",
                                                target_module=Linear1D_Row,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=Dropout1D,
                                            )
                                        ])
        }

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model


class T5ForConditionalGenerationPolicy(T5ModelPolicy):

    def module_policy(self):
        policy = super().module_policy()

        new_item = {
            T5ForConditionalGeneration:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(suffix="lm_head",
                                                                            target_module=Linear1D_Col,
                                                                            kwargs=dict(gather_output=True))
                                        ])
        }

        policy.update(new_item)
        return policy


class T5EncoderPolicy(T5ModelPolicy):
    pass
