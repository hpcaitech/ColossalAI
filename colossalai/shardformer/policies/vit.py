import warnings
from typing import Callable, Dict, List, Union

import torch.nn as nn

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.layer import DropoutForReplicatedInput, Linear1D_Col

from ..modeling.jit import get_jit_fused_dropout_add_func
from ..modeling.vit import (
    ViTForImageClassification_pipeline_forward,
    ViTForMaskedImageModeling_pipeline_forward,
    ViTModel_pipeline_forward,
    get_jit_fused_vit_intermediate_forward,
    get_jit_fused_vit_output_forward,
    get_vit_flash_self_attention_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["ViTPolicy", "ViTModelPolicy", "ViTForImageClassificationPolicy", "ViTForMaskedImageModelingPolicy"]


class ViTPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        self.enable_bias_gelu_fused = self.shard_config.enable_jit_fused and self.model.config.hidden_act == "gelu"
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.vit.modeling_vit import (
            ViTEmbeddings,
            ViTIntermediate,
            ViTLayer,
            ViTOutput,
            ViTSelfAttention,
        )

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Vit doesn't support sequence parallelism now, will ignore the sequence parallelism flag.")

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), f"The number of attention heads must be divisible by tensor parallel size."
            policy[ViTEmbeddings] = ModulePolicyDescription(
                attribute_replacement={},
                param_replacement=[],
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForReplicatedInput,
                    )
                ],
            )

            policy[ViTLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.attention.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "attention.attention.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
                param_replacement=[],
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.attention.query",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.key",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.value",
                        target_module=col_nn.Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate.dense",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "skip_bias_add": self.enable_bias_gelu_fused,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dense",
                        target_module=col_nn.Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ],
            )
            if self.enable_bias_gelu_fused:
                self.append_or_create_method_replacement(
                    description={
                        "forward": get_jit_fused_vit_intermediate_forward(),
                    },
                    policy=policy,
                    target_key=ViTIntermediate,
                )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_vit_flash_self_attention_forward(),
                },
                policy=policy,
                target_key=ViTSelfAttention,
            )

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_vit_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=ViTOutput,
            )

        return policy

    def new_model_class(self):
        return None

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        if self.model.__class__.__name__ == "ViTModel":
            module = self.model
        else:
            module = self.model.vit
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, pipeline_forward: Callable, policy: Dict):
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "ViTModel":
                module = self.model
            else:
                module = self.model.vit

            layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {"forward": pipeline_forward(stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )


# ViTModel
class ViTModelPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTModel

        policy = super().module_policy()

        if self.shard_config.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=ViTModel, pipeline_forward=ViTModel_pipeline_forward, policy=policy)
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)
            held_layers.append(module.pooler)

        return held_layers


# ViTForImageClassification
class ViTForImageClassificationPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTModel

        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            new_item = {
                ViTForImageClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="classifier", target_module=Linear1D_Col, kwargs=dict(gather_output=True)
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.shard_config.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=ViTModel, pipeline_forward=ViTModel_pipeline_forward, policy=policy)
            self.set_pipeline_forward(
                model_cls=ViTForImageClassification,
                pipeline_forward=ViTForImageClassification_pipeline_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        module = self.model.vit
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)
            held_layers.append(self.model.classifier)

        return held_layers


# ViTForMaskedImageModeling
class ViTForMaskedImageModelingPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTForMaskedImageModeling, ViTModel

        policy = super().module_policy()

        if self.shard_config.pipeline_stage_manager is not None:
            self.set_pipeline_forward(model_cls=ViTModel, pipeline_forward=ViTModel_pipeline_forward, policy=policy)
            self.set_pipeline_forward(
                model_cls=ViTForMaskedImageModeling,
                pipeline_forward=ViTForMaskedImageModeling_pipeline_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        assert self.pipeline_stage_manager is not None, "pipeline_stage_manager is None"

        module = self.model.vit
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)
            held_layers.append(self.model.decoder)

        return held_layers
