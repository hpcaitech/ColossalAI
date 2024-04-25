import torch.nn as nn
from torch.nn import Parameter

from colossalai.inference.modeling.models.nopadding_baichuan import (
    NopadBaichuanAttention,
    NopadBaichuanMLP,
    baichuan_rmsnorm_forward,
)
from colossalai.inference.modeling.models.nopadding_llama import (
    llama_causal_lm_forward,
    llama_decoder_layer_forward,
    llama_model_forward,
)
from colossalai.inference.utils import init_to_get_rotary
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class NoPaddingBaichuanModelInferPolicy(LlamaForCausalLMPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()

        decoder_attribute_replacement = {
            "lm_head.weight": Parameter(nn.functional.normalize(self.model.lm_head.weight), requires_grad=False),
        }
        policy["BaichuanForCausalLM"] = ModulePolicyDescription(
            attribute_replacement=decoder_attribute_replacement,
        )

        # used for relpacing Baichuan 7B/13B decoder layer
        for layer_name in ["DecoderLayer", "BaichuanLayer"]:
            policy[layer_name] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="mlp",
                        target_module=NopadBaichuanMLP,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn",
                        target_module=NopadBaichuanAttention,
                    ),
                ]
            )

            self.append_or_create_method_replacement(
                description={"forward": llama_decoder_layer_forward}, policy=policy, target_key=layer_name
            )

        self.append_or_create_method_replacement(
            description={"forward": llama_causal_lm_forward}, policy=policy, target_key="BaichuanForCausalLM"
        )
        self.append_or_create_method_replacement(
            description={"forward": llama_model_forward}, policy=policy, target_key="BaichuanModel"
        )

        self.append_or_create_method_replacement(
            description={"forward": baichuan_rmsnorm_forward}, policy=policy, target_key="RMSNorm"
        )

        return policy

    def postprocess(self):
        init_to_get_rotary(self.model.model)
        return self.model
