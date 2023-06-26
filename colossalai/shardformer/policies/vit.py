from typing import Dict, Union

import torch.nn as nn
from transformers.models.vit.modeling_vit import ViTAttention, ViTEmbeddings, ViTLayer, ViTModel

from colossalai.shardformer.layer import DropoutForReplicatedInput, Linear1D_Col, Linear1D_Row

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class ViTPolicy(Policy):

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
            ViTEmbeddings:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
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
                                        param_replacement=[],
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

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model
