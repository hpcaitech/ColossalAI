from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel

from colossalai.inference.modeling.models.glide_llama import (
    GlideLlamaDecoderLayer,
    glide_llama_causal_lm_forward,
    glide_llama_model_forward,
)
from colossalai.inference.utils import init_to_get_rotary
from colossalai.shardformer.policies.base_policy import SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy


class GlideLlamaModelPolicy(LlamaForCausalLMPolicy):
    def module_policy(self):
        policy = super().module_policy()

        num_layers = self.model.config.num_hidden_layers
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix=f"layers[{i}]",
                    target_module=GlideLlamaDecoderLayer,
                )
                for i in range(num_layers)
            ],
            policy=policy,
            target_key=LlamaModel,
        )
        self.append_or_create_method_replacement(
            description={"forward": glide_llama_model_forward},
            policy=policy,
            target_key=LlamaModel,
        )
        self.append_or_create_method_replacement(
            description={"forward": glide_llama_causal_lm_forward},
            policy=policy,
            target_key=LlamaForCausalLM,
        )

        return policy

    def postprocess(self):
        for layer in self.model.model.layers:
            init_to_get_rotary(layer.cross_attn)
        return self.model
