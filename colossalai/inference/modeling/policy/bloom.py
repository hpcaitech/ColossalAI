from torch.nn import Parameter
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomModel

from colossalai.inference.modeling.models.bloom import (
    bloom_causal_lm_forward,
    bloom_model_forward,
)
from colossalai.inference.utils import init_to_get_rotary
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.bloom import BloomForCausalLMPolicy

class BloomModelInferPolicy(BloomForCausalLMPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        decoder_attribute_replacement = {
            "lm_head.weight": Parameter(self.model.lm_head.weight.transpose(0, 1), requires_grad=False),
        }
        policy[BloomForCausalLM] = ModulePolicyDescription(
            attribute_replacement=decoder_attribute_replacement,
        )

        self.append_or_create_method_replacement(
            description={"forward": bloom_causal_lm_forward}, policy=policy, target_key=BloomForCausalLM
        )
        self.append_or_create_method_replacement(
            description={"forward": bloom_model_forward}, policy=policy, target_key=BloomModel
        )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model)
        return self.model