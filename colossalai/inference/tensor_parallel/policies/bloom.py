from colossalai.shardformer.policies.bloom import BloomForCausalLMPolicy

from ..modeling.bloom import BloomInferenceForwards


class BloomModelInferPolicy(BloomForCausalLMPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomForCausalLM, BloomModel
        policy = super().module_policy()
        # NOTE set inference mode to shard config
        self.shard_config._infer()

        if self.shard_config.enable_tensor_parallelism:

            method_replacement = {
                'forward':
                    BloomInferenceForwards.bloom_for_causal_lm_forward,
                'prepare_inputs_for_generation':
                    BloomInferenceForwards.bloom_for_causal_lm_prepare_inputs_for_generation
            }
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=BloomForCausalLM)

            method_replacement = {'forward': BloomInferenceForwards.bloom_model_forward}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=BloomModel)

            method_replacement = {'forward': BloomInferenceForwards.bloom_block_forward}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=BloomBlock)

            method_replacement = {'forward': BloomInferenceForwards.bloom_attention_forward}
            self.append_or_create_method_replacement(description=method_replacement,
                                                     policy=policy,
                                                     target_key=BloomAttention)

        return policy
