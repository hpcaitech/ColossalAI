from transformers.models.llama.modeling_llama import LlamaModel

from colossalai.inference.modeling.models.glide_llama import GlideLlamaDecoderLayer, glide_llama_model_forward
from colossalai.shardformer.policies.base_policy import SubModuleReplacementDescription
from colossalai.shardformer.policies.llama import LlamaModelPolicy


# class GlideLlamaModelPolicy(LlamaForCausalLMPolicy):
class GlideLlamaModelPolicy(LlamaModelPolicy):
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

        return policy
