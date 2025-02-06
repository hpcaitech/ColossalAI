from typing import Dict, Union

import torch.nn as nn

from colossalai.shardformer.layer import FusedRMSNorm
from colossalai.shardformer.modeling.deepseek_v3 import EpDeepseekV3MoE
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = ["DeepseekPolicy", "DeepseekForCausalLMPolicy"]


class DeepseekV3Policy(Policy):
    def config_sanity_check(self):
        assert not self.shard_config.enable_tensor_parallelism, "DeepSeekV3 does not support tensor parallelism"
        assert self.shard_config.pipeline_stage_manager is None, "DeepSeekV3 does not support pipeline parallelism"
        assert not self.shard_config.enable_sequence_parallelism, "DeepSeekV3 does not support sequence parallelism"

    def preprocess(self):
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:

        policy = {}

        if self.shard_config.ep_group:
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


class DeepseekV3ModelPolicy(DeepseekV3Policy):
    pass


class DeepseekV3ForCausalLMPolicy(DeepseekV3Policy):
    pass
