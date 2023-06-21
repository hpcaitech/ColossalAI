from typing import Dict, Union

import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaForSequenceClassification
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel

from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class LlamaPolicy(Policy):

    def preprocess(self):
        # Resize embedding
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size

        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        return {
            LlamaDecoderLayer:
                ModulePolicyDescription(
                    attribute_replacement={
                        "self_attn.hidden_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                        "self_attn.num_heads":
                            self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                    },
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="self_attn.q_proj",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="self_attn.k_proj",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="self_attn.v_proj",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="self_attn.o_proj",
                            target_module=Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="mlp.gate_proj",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="mlp.up_proj",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="mlp.down_proj",
                            target_module=Linear1D_Row,
                        )
                    ],
                ),
            LlamaModel:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="embed_tokens",
                                                target_module=VocabParallelEmbedding1D,
                                            )
                                        ])
        }

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model


class LlamaForCausalLMPolicy(LlamaPolicy):

    def module_policy(self):
        policy = super().module_policy()
        # add a new item for casual lm
        new_item = {
            LlamaForCausalLM:
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


class LlamaForSequenceClassificationPolicy(LlamaPolicy):

    def module_policy(self):
        policy = super().module_policy()

        # add a new item for sequence classification
        new_item = {
            LlamaForSequenceClassification:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(suffix="score",
                                                                            target_module=Linear1D_Col,
                                                                            kwargs=dict(gather_output=True))
                                        ])
        }
        policy.update(new_item)
        return policy
