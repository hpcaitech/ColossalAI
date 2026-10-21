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
    get_jit_fused_vit_layer_forward,
    get_vit_flash_self_attention_forward,
)
from .base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["ViTPolicy", "ViTModelPolicy", "ViTForImageClassificationPolicy", "ViTForMaskedImageModelingPolicy"]


class ViTPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        from transformers.models.vit.modeling_vit import ViTAttention, ViTEmbeddings, ViTLayer

        policy = {}
        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("ViT does not support sequence parallelism; the flag is disabled.")
        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv
        tp = self.shard_config.enable_tensor_parallelism
        if tp or use_zbv:
            if tp:
                assert self.model.config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            policy[ViTEmbeddings] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(suffix="dropout", target_module=DropoutForReplicatedInput)
                ]
            )
            replacements = []
            for suffix in (
                "attention.q_proj",
                "attention.k_proj",
                "attention.v_proj",
                "attention.o_proj",
                "mlp.fc1",
                "mlp.fc2",
            ):
                column = suffix in ("attention.q_proj", "attention.k_proj", "attention.v_proj", "mlp.fc1")
                target = (col_nn.Linear1D_Col if column else col_nn.Linear1D_Row) if tp else col_nn.LinearWithGradAccum
                replacements.append(
                    SubModuleReplacementDescription(
                        suffix=suffix,
                        target_module=target,
                        kwargs={"fp8_communication": self.shard_config.fp8_communication, "use_zbv": use_zbv},
                    )
                )
            replacements.append(
                SubModuleReplacementDescription(suffix="dropout", target_module=col_nn.DropoutForReplicatedInput)
            )
            policy[ViTLayer] = ModulePolicyDescription(
                attribute_replacement=(
                    {
                        "attention.num_attention_heads": self.model.config.num_attention_heads
                        // self.shard_config.tensor_parallel_size
                    }
                    if tp
                    else {}
                ),
                sub_module_replacement=replacements,
            )
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={"forward": get_vit_flash_self_attention_forward()},
                policy=policy,
                target_key=ViTAttention,
            )
        if self.shard_config.enable_jit_fused:
            # HF's default gelu is exact erf GELU. The legacy bias_gelu kernel is a
            # tanh approximation: retain the configured MLP activation and fuse
            # dropout/residual at the new ViTLayer boundary instead.
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_vit_layer_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=ViTLayer,
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
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embeddings)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.layers[start_idx:end_idx])
        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            if stage_manager.is_first_stage():
                held_layers.append(module.embeddings)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.layers[start_idx:end_idx])
        return held_layers

    def set_pipeline_forward(self, model_cls: nn.Module, pipeline_forward: Callable, policy: Dict):
        if self.pipeline_stage_manager:
            stage_manager = self.pipeline_stage_manager
            if self.model.__class__.__name__ == "ViTModel":
                module = self.model
            else:
                module = self.model.vit

            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
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
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.layernorm)
                held_layers.append(module.pooler)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(module.layernorm)
                held_layers.append(module.pooler)

        return held_layers


# ViTForImageClassification
class ViTForImageClassificationPolicy(ViTPolicy):
    def module_policy(self):
        from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTModel

        policy = super().module_policy()

        use_zbv = self.pipeline_stage_manager is not None and self.pipeline_stage_manager.use_zbv

        if self.shard_config.enable_tensor_parallelism:
            new_item = {
                ViTForImageClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="classifier",
                            target_module=Linear1D_Col,
                            kwargs=dict(
                                gather_output=True,
                                fp8_communication=self.shard_config.fp8_communication,
                                use_zbv=use_zbv,
                            ),
                        )
                    ]
                )
            }
            policy.update(new_item)
        elif use_zbv:
            new_item = {
                ViTForImageClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="classifier",
                            target_module=col_nn.LinearWithGradAccum,
                            kwargs=dict(
                                gather_output=True,
                                fp8_communication=self.shard_config.fp8_communication,
                                use_zbv=use_zbv,
                            ),
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
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.layernorm)
                held_layers.append(self.model.classifier)
        else:
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
        if stage_manager.is_interleave:
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                held_layers.append(module.layernorm)
                held_layers.append(self.model.decoder)
        else:
            if stage_manager.is_last_stage():
                held_layers.append(module.layernorm)
                held_layers.append(self.model.decoder)

        return held_layers
