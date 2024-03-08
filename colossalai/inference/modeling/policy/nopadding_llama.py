from functools import partial

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

# import colossalai
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

        self.shard_config._infer()

        infer_forward = llama_causal_lm_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=LlamaForCausalLM
        )

        infer_forward = llama_model_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaModel)

        infer_forward = llama_decoder_layer_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=LlamaDecoderLayer
        )

        infer_forward = llama_rmsnorm_forward
        method_replacement = {"forward": partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaRMSNorm)

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model)
        return self.model
