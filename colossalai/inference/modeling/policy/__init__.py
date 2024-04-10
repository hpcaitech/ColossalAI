from .glide_llama import GlideLlamaModelPolicy
from .nopadding_llama import NoPaddingLlamaModelInferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "glide_llama": GlideLlamaModelPolicy,
}

__all__ = ["NoPaddingLlamaModelInferPolicy", "GlideLlamaModelPolicy", "model_polic_map"]
