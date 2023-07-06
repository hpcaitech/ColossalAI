from typing import Dict, Union

import torch.nn as nn

from colossalai.shardformer.layer import DropoutForReplicatedInput, DropoutForParallelInput, FusedLayerNorm, Linear1D_Col, Linear1D_Row

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ['ViTPolicy', 'ViTForImageClassificationPolicy', 'ViTForMaskedImageModelingPolicy']


class ViTPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer

        policy = {}

        if self.shard_config.enable_tensor_parallelism:
            policy[ViTEmbeddings] = ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="dropout",
                                                target_module=DropoutForReplicatedInput,
                                            )
                                        ])
            
            policy[ViTLayer] = ModulePolicyDescription(
                    attribute_replacement={
                        "attention.attention.num_attention_heads":
                            self.model.config.num_attention_heads//self.shard_config.tensor_parallel_size,
                        "attention.attention.all_head_size":
                            self.model.config.hidden_size//self.shard_config.tensor_parallel_size,
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
                            target_module=DropoutForReplicatedInput,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.output.dense",
                            target_module=Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.output.dropout",
                            target_module=DropoutForReplicatedInput,
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
                            target_module=DropoutForReplicatedInput,
                        ),
                    ]
                )

        return policy
  
    
    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model

class ViTForImageClassificationPolicy(ViTPolicy):

     def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTForImageClassification

        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            new_item = {
                ViTForImageClassification:
                ModulePolicyDescription(sub_module_replacement=[
                                        SubModuleReplacementDescription(suffix="classifier",
                                                                            target_module=Linear1D_Col,
                                                                            kwargs=dict(gather_output=True))
                                        ])
            }
            policy.update(new_item)
        return policy

class ViTForMaskedImageModelingPolicy(ViTPolicy):
    
    def module_policy(self):
        policy = super().module_policy()
        return policy
    

        

