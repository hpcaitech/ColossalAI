from functools import partial

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaRMSNorm

from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

from ..modeling.llama import LlamaInferenceForwards, get_llama_vllm_rmsnorm_forward


class LlamaModelInferPolicy(LlamaForCausalLMPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        policy = super().module_policy()
        self.shard_config._infer()

        infer_forward = LlamaInferenceForwards.llama_model_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=LlamaModel)

        infer_forward = LlamaInferenceForwards.llama_decoder_layer_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=policy,
                                                 target_key=LlamaDecoderLayer)

        infer_forward = LlamaInferenceForwards.llama_flash_attn_kvcache_forward
        method_replacement = {'forward': partial(infer_forward)}
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=policy,
                                                 target_key=LlamaAttention)

        # NOTE: adding rms_norm caused precision issue, fix @tiandiao123
        # infer_forward = get_llama_vllm_rmsnorm_forward()
        # if infer_forward is not None:
        #     method_replacement = {'forward': partial(infer_forward)}
        #     self.append_or_create_method_replacement(description=method_replacement,
        #                                              policy=policy,
        #                                              target_key=LlamaRMSNorm)

        return policy
