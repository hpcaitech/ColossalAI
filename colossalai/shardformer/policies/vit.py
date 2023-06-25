rom typing import Dict, Union

import torch.nn as nn

from transformers.models.vit.modeling_vit import ViTModel, ViTLayer, ViTEmbeddings, ViTAttention

from colossalai.shardformer.layer.layers import Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D, LayerNorm1D, Dropout1D

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
        return  {
            ViTEmbeddings:
                ModulePolicyDescription(
                    attribute_replacement{},
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=Dropout1D,
                        )
                    ]
                ),
            ViTLayer:
                ModulePolicyDescription(
                    attribute_replacement{},
                    param_replacement=[],
                    sub_module_replacement=[
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
                            target_module=Dropout1D,
                        ),
                        SubModuleReplacementDescription(
                            suffix="layernorm_before",
                            target_module=LayerNorm1D,
                        ),
                        SubModuleReplacementDescription(
                            suffix="layernorm_after",
                            target_module=LayerNorm1D,
                        ),
                    ]
                ),
            ViTAttention:
                ModulePolicyDescription(
                    attribute_replacement{
                        "attention.num_attention_heads":
                            self.config.num_attention_heads//self.shard_config.tensor_parallel_size,
                            
                    },
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="attention.query",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.key",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.value",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.dropout",
                            target_module=Dropout1D,
                        ),
                        SubModuleReplacementDescription(
                            suffix="output.dense",
                            target_module=Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="output.dropout",
                            target_module=Dropout1D,
                        ),
                    ],
                ),
            ViTModel:
                ModulePolicyDescription(
                    attribute_replacement{},
                    param_replacement=[],
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="layernorm",
                            target_module=LayerNorm1D,
                        )
                    ]
                ),
        }





