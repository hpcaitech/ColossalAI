from typing import Dict, Union

import torch.nn as nn

from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row, VocabParallelEmbedding1D
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class Grok1Policy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        if self.shard_config.enable_tensor_parallelism:
            vocab_size = self.model.config.vocab_size
            world_size = self.shard_config.tensor_parallel_size
            assert vocab_size % world_size == 0, f"vocab_size {vocab_size} must be divisible by world_size {world_size}"
        return self.model

    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = {}
        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "attn.hidden_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                "attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                "attn.num_key_value_heads": self.model.config.num_key_value_heads
                // self.shard_config.tensor_parallel_size,
            }
            decoder_submodule_replacement = [
                SubModuleReplacementDescription(
                    suffix="attn.q_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="attn.k_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="attn.v_proj",
                    target_module=Linear1D_Col,
                ),
                SubModuleReplacementDescription(
                    suffix="attn.o_proj",
                    target_module=Linear1D_Row,
                ),
            ]
            for i in range(self.model.config.num_experts):
                decoder_submodule_replacement.extend(
                    [
                        SubModuleReplacementDescription(
                            suffix=f"moe_block.experts[{i}].linear",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix=f"moe_block.experts[{i}].linear_v",
                            target_module=Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix=f"moe_block.experts[{i}].linear_1",
                            target_module=Linear1D_Row,
                        ),
                    ]
                )

            policy["DecoderLayer"] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=decoder_submodule_replacement,
            )
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                ),
                policy=policy,
                target_key="Grok1Model",
            )
        return policy

    def postprocess(self):
        return self.model


class Grok1ModelPolicy(Grok1Policy):
    pass


class Grok1ForCausalLMPolicy(Grok1Policy):
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        policy = super().module_policy()
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="lm_head",
                target_module=Linear1D_Col,
                kwargs={"gather_output": not self.shard_config.parallel_output},
            ),
            policy=policy,
            target_key="Grok1ModelForCausalLM",
        )
        return policy
