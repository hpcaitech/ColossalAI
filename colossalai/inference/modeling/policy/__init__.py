from .glide_llama import GlideLlamaModelPolicy
from .nopadding_baichuan import NoPaddingBaichuanModelInferPolicy
from .nopadding_llama import NoPaddingLlamaModelInferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "nopadding_baichuan": NoPaddingBaichuanModelInferPolicy,
    "glide_llama": GlideLlamaModelPolicy,
}

__all__ = [
    "NoPaddingLlamaModelInferPolicy",
    "NoPaddingBaichuanModelInferPolicy",
    "GlideLlamaModelPolicy",
    "model_polic_map",
]
