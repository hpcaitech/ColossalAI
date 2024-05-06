from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomModel

from colossalai.inference.modeling.models.nopadding_bloom import (
    bloom_attention_forward,
    bloom_block_forward,
    bloom_causal_lm_forward,
    bloom_model_forward,
)
from colossalai.shardformer.policies.bloom import BloomForCausalLMPolicy


class NoPaddingBloomModelInferPolicy(BloomForCausalLMPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        # policy[BloomBlock] = ModulePolicyDescription(
        #     sub_module_replacement=[
        #         SubModuleReplacementDescription(
        #             suffix="mlp",
        #             target_module=NopadBloomMLP,
        #         ),
        #         # SubModuleReplacementDescription(
        #         #     suffix="self_attention",
        #         #     target_module=NopadBloomAttention,
        #         # ),
        #     ]
        # )

        self.append_or_create_method_replacement(
            description={"forward": bloom_causal_lm_forward},
            policy=policy,
            target_key=BloomForCausalLM,
        )
        self.append_or_create_method_replacement(
            description={"forward": bloom_model_forward},
            policy=policy,
            target_key=BloomModel,
        )
        self.append_or_create_method_replacement(
            description={"forward": bloom_block_forward},
            policy=policy,
            target_key=BloomBlock,
        )
        self.append_or_create_method_replacement(
            description={"forward": bloom_attention_forward},
            policy=policy,
            target_key=BloomAttention,
        )

        return policy

    def postprocess(self):
        return self.model
