from .nopadding_llama import NoPaddingLlamaModelInferPolicy

# from colossalai.shardformer.policies.llama import LlamaForCausalLMPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
}

__all__ = ["NoPaddingLlamaModelInferPolicy", "model_policy_map"]
