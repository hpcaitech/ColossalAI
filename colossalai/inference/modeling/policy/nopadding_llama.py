from torch.nn import Parameter
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel, LlamaRMSNorm

from colossalai.inference.modeling.models.nopadding_llama import (
    NopadLlamaAttention,
    NopadLlamaMLP,
    llama_causal_lm_forward,
    llama_decoder_layer_forward,
    llama_model_forward,
    llama_rmsnorm_forward,
)
from colossalai.inference.utils import init_to_get_rotary
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class NoPaddingLlamaModelInferPolicy(LlamaForCausalLMPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        decoder_attribute_replacement = {
            "lm_head.weight": Parameter(self.model.lm_head.weight.transpose(0, 1), requires_grad=False),
        }
        policy[LlamaForCausalLM] = ModulePolicyDescription(
            attribute_replacement=decoder_attribute_replacement,
        )

        policy[LlamaDecoderLayer] = ModulePolicyDescription(
            sub_module_replacement=[
                SubModuleReplacementDescription(
                    suffix="mlp",
                    target_module=NopadLlamaMLP,
                ),
                SubModuleReplacementDescription(
                    suffix="self_attn",
                    target_module=NopadLlamaAttention,
                ),
            ]
        )

        self.append_or_create_method_replacement(
            description={"forward": llama_causal_lm_forward}, policy=policy, target_key=LlamaForCausalLM
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_model_forward}, policy=policy, target_key=LlamaModel
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_decoder_layer_forward}, policy=policy, target_key=LlamaDecoderLayer
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_rmsnorm_forward}, policy=policy, target_key=LlamaRMSNorm
        )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model)
        return self.model
