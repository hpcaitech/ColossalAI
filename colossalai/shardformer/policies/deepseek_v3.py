from functools import partial
from typing import Callable, Dict, List, Union

import torch.nn as nn

from colossalai.shardformer.layer import FusedRMSNorm
from colossalai.shardformer.modeling.deepseek_v3 import (
    EpDeepseekV3MoE,
    deepseek_v3_for_causal_lm_forward,
    deepseek_v3_model_forward,
)
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["DeepseekPolicy", "DeepseekForCausalLMPolicy"]


class DeepseekV3Policy(Policy):
    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism, "DeepSeekV3 does not support tensor parallelism"
        assert not self.shard_config.enable_sequence_parallelism, "DeepSeekV3 does not support sequence parallelism"
        if self.shard_config.pipeline_stage_manager:
            assert not self.shard_config.pipeline_stage_manager.use_zbv, "DeepSeekV3 does not support ZBV"

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:

        policy = {}

        # support gradient checkpointing
        if self.shard_config.pipeline_stage_manager is None:
            policy["DeepseekV3Model"] = ModulePolicyDescription(
                method_replacement={"forward": deepseek_v3_model_forward}
            )

        if self.shard_config.expert_parallel_size > 1:
            # expert parallel
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="mlp",
                        target_module=EpDeepseekV3MoE,
                        kwargs={
                            "ep_group": self.shard_config.ep_group,
                            "moe_dp_group": self.shard_config.moe_dp_group,
                        },
                    )
                ],
                policy=policy,
                target_key="DeepseekV3DecoderLayer",
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            # TODO: prevent casting to fp32
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
                target_key="DeepseekV3DecoderLayer",
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key="DeepseekV3Model",
            )

        return policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(self, model_cls: str, new_forward: Callable, policy: Dict) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager:
            num_layers = self.model.config.num_hidden_layers
            stage_manager = self.pipeline_stage_manager

            layers_per_stage = stage_manager.distribute_layers(num_layers)
            stage_index = stage_manager.get_stage_index(layers_per_stage)
            method_replacement = {"forward": partial(new_forward, stage_manager=stage_manager, stage_index=stage_index)}
            self.append_or_create_method_replacement(
                description=method_replacement, policy=policy, target_key=model_cls
            )

        return

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        module = self.model
        if module.__class__.__name__.startswith("PeftModel"):
            module = module.get_base_model()
        if module.__class__.__name__ != "DeepseekV3Model":
            module = module.model

        stage_manager = self.pipeline_stage_manager

        held_layers = []

        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            stage_indices = stage_manager.get_stage_index(layers_per_stage)
            stage_manager.stage_indices = stage_indices
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embed_tokens)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.layers[start_idx:end_idx])
            if (stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True)) or (
                not stage_manager.use_zbv and stage_manager.is_last_stage(ignore_chunk=True)
            ):
                # for zbv, when is_first_stage (last fwd), we append norm
                # for interleaved, when is_last_stage (last fwd), we also append norm
                held_layers.append(module.norm)
        else:
            layers_per_stage = stage_manager.distribute_layers(len(module.layers))
            if stage_manager.is_first_stage():
                held_layers.append(module.embed_tokens)
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
            held_layers.extend(module.layers[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.norm)
        return held_layers


class DeepseekV3ModelPolicy(DeepseekV3Policy):
    def module_policy(self):
        policy = super().module_policy()
        if self.shard_config.pipeline_stage_manager:
            self.set_pipeline_forward("DeepseekV3Model", deepseek_v3_model_forward, policy)
        return policy


class DeepseekV3ForCausalLMPolicy(DeepseekV3Policy):
    def module_policy(self):
        policy = super().module_policy()
        if self.shard_config.pipeline_stage_manager:
            self.set_pipeline_forward("DeepseekV3ForCausalLM", deepseek_v3_for_causal_lm_forward, policy)
        return policy

    def get_held_layers(self):
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.use_zbv and stage_manager.is_first_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        elif stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        return held_layers
