from .glide_llama import GlideLlamaModelPolicy
from .nopadding_baichuan import NoPaddingBaiChuanModelInferPolicy
from .nopadding_llama import NoPaddingLlamaModelInferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "nopadding_baichuan": NoPaddingBaiChuanModelInferPolicy,
    "glide_llama": GlideLlamaModelPolicy,
}

__all__ = [
    "NoPaddingLlamaModelInferPolicy",
    "NoPaddingBaiChuanModelInferPolicy",
    "GlideLlamaModelPolicy",
    "model_polic_map",
]
