from .glide_llama import GlideLlamaModelPolicy
from .nopadding_baichuan import NoPaddingBaichuanModelInferPolicy
from .nopadding_llama import NoPaddingLlamaModelInferPolicy
from .nopadding_bloom import NoPaddingBloomModelInferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "nopadding_baichuan": NoPaddingBaichuanModelInferPolicy,
    "nopadding_bloom": NoPaddingBloomModelInferPolicy,
    "glide_llama": GlideLlamaModelPolicy,
}

__all__ = [
    "NoPaddingLlamaModelInferPolicy",
    "NoPaddingBaichuanModelInferPolicy",
    "GlideLlamaModelPolicy",
    "NoPaddingBloomModelInferPolicy",
    "model_polic_map",
]