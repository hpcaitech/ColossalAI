from .nopadding_llama import NoPaddingLlamaModelInferPolicy
from .padding_llama import PaddingLlamaModelInferPolicy

model_policy_map = {
    "padding_llama": PaddingLlamaModelInferPolicy,
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
}

__all__ = ["PaddingLlamaModelInferPolicy", "NoPaddingLlamaModelInferPolicy", "model_polic_map"]
