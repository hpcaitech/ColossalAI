from functools import partial

from .modeling.llama import LlamaInferenceForwards
from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

class LlamaModelInferPolicy(LlamaForCausalLMPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel
        policy = super().module_policy()
        self.shard_config._infer()

        # example for replace layer or decoder
        # if self.shard_config.enable_flash_attention:
        #     policy[LlamaAttention] = ModulePolicyDescription(method_replacement={
        #         'forward': get_llama_flash_attention_forward(),
        #     })

        infer_forward = LlamaInferenceForwards.llama_model_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaModel)
        
        infer_forward = LlamaInferenceForwards.llama_decoder_layer_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaDecoderLayer)
        
        infer_forward = LlamaInferenceForwards.llama_flash_attn_kvcache_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaAttention)

        return policy