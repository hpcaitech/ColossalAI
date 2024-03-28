import warnings
from typing import Dict, Union

import torch.nn as nn

from colossalai.shardformer.layer import FusedRMSNorm, Linear1D_Col, VocabParallelLMHead1D, PaddingLMHead, Linear1D_Row, VocabParallelEmbedding1D, PaddingEmbedding

from ..modeling.mistral import get_mistral_flash_attention_forward
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["MistralPolicy", "MistralModelPolicy", "MistralForCausalLMPolicy", "MistralForSequenceClassificationPolicy"]


class MistralPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model
    
    def tie_weight_check(self):
        input_embedding = self.model.get_input_embeddings()
        output_embedding = self.model.get_output_embeddings()
        return input_embedding is not None and output_embedding is not None and id(input_embedding.weight) == id(output_embedding.weight)

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.mistral.modeling_mistral import MistralAttention, MistralDecoderLayer, MistralModel

        policy = {}

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        else:
            if self.tie_weight_check():
                embedding_cls = PaddingEmbedding

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Mistral doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                "self_attn.num_key_value_heads": self.model.config.num_key_value_heads
                // self.shard_config.tensor_parallel_size,
            }

            policy[MistralDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
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
                    ),
                ],
            )

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs={"make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by}
                ),
                policy=policy,
                target_key=MistralModel,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="input_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=MistralDecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=MistralModel,
            )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_mistral_flash_attention_forward(),
                },
                policy=policy,
                target_key=MistralAttention,
            )

        return policy

    def postprocess(self):
        return self.model


class MistralModelPolicy(MistralPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        if self.pipeline_stage_manager:
            warnings.warn("Mistral doesn't support pipeline parallelism now.")

        return super().module_policy()


class MistralForCausalLMPolicy(MistralPolicy):
    def module_policy(self):
        from transformers import MistralForCausalLM

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            warnings.warn("Mistral doesn't support pipeline parallelism now.")

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                MistralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=VocabParallelLMHead1D, kwargs=dict(gather_output=True, make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by)
                        )
                    ]
                )
            }
        else:
            new_item = {
                MistralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head", target_module=PaddingLMHead, kwargs=dict(make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by)
                        )
                    ]
                )
            }

        policy.update(new_item)

        return policy


class MistralForSequenceClassificationPolicy(MistralPolicy):
    def module_policy(self):
        from transformers import MistralForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                MistralForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }

            if self.pipeline_stage_manager:
                warnings.warn("Mistral doesn't support pipeline parallelism now.")

            policy.update(new_item)
        return policy
