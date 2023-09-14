from functools import partial

import torch
from torch.nn import LayerNorm

from colossalai.shardformer.policies.bloom import BloomForCausalLMPolicy

from ..modeling.bloom import BloomInferenceForwards

try:
    from colossalai.kernel.triton import layer_norm
    HAS_TRITON_NORM = True
except:
    print("Some of our kernels require triton. You might want to install triton from https://github.com/openai/triton")
    HAS_TRITON_NORM = False


def get_triton_layernorm_forward():
    if HAS_TRITON_NORM:

        def _triton_layernorm_forward(self: LayerNorm, hidden_states: torch.Tensor):
            return layer_norm(hidden_states, self.weight.data, self.bias, self.eps)

        return _triton_layernorm_forward
    else:
        return None


class BloomModelInferPolicy(BloomForCausalLMPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomModel
        policy = super().module_policy()
        # NOTE set inference mode to shard config
        self.shard_config._infer()

        method_replacement = {
            'forward': BloomInferenceForwards.bloom_for_causal_lm_forward,
            'prepare_inputs_for_generation': BloomInferenceForwards.bloom_for_causal_lm_prepare_inputs_for_generation
        }
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=policy,
                                                 target_key=BloomForCausalLM)

        method_replacement = {'forward': BloomInferenceForwards.bloom_model_forward}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=BloomModel)

        method_replacement = {'forward': BloomInferenceForwards.bloom_block_forward}
        self.append_or_create_method_replacement(description=method_replacement, policy=policy, target_key=BloomBlock)

        method_replacement = {'forward': BloomInferenceForwards.bloom_attention_forward}
        self.append_or_create_method_replacement(description=method_replacement,
                                                 policy=policy,
                                                 target_key=BloomAttention)

        if HAS_TRITON_NORM:
            infer_method = get_triton_layernorm_forward()
            method_replacement = {'forward': partial(infer_method)}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=LayerNorm)

        return policy
