from .nopadding_llama import NoPaddingLlamaModelInferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
}

__all__ = ["NoPaddingLlamaModelInferPolicy", "model_polic_map"]
