from typing import Dict, Union

import torch.nn as nn

from colossalai.shardformer.layer import DropoutForReplicatedInput, FusedLayerNorm, Linear1D_Col, Linear1D_Row

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ['ViTPolicy']


class ViTPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # Resize embedding
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size

        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)

        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer

        base_policy = {
            ViTEmbeddings:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForReplicatedInput,
                    )
                ]),
            ViTLayer:
                ModulePolicyDescription(attribute_replacement={
                    "attention.attention.num_attention_heads":
                        self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                    "attention.attention.all_head_size":
                        self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                },
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="attention.attention.query",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attention.attention.key",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attention.attention.value",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attention.attention.dropout",
                                                target_module=DropoutForParallelInput,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attention.output.dense",
                                                target_module=Linear1D_Row,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attention.output.dropout",
                                                target_module=DropoutForParallelInput,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="intermediate.dense",
                                                target_module=Linear1D_Col,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="output.dense",
                                                target_module=Linear1D_Row,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="output.dropout",
                                                target_module=DropoutForParallelInput,
                                            ),
                                        ]),
        }

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            base_policy[ViTAttention].sub_module_replacement.extend([
                SubModuleReplacementDescription(
                    suffix="layernorm_before",
                    target_module=FusedLayerNorm,
                ),
                SubModuleReplacementDescription(
                    suffix="layernorm_after",
                    target_module=FusedLayerNorm,
                )
            ])
            base_policy[ViTModel].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="layernorm",
                    target_module=FusedLayerNorm,
                ))

        return base_policy

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model
