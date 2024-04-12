from .nopadding_baichuan import NoPaddingBaiChuanModelInferPolicy
from .nopadding_llama import NoPaddingLlamaModelInferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "nopadding_baichuan": NoPaddingBaiChuanModelInferPolicy,
}

__all__ = ["NoPaddingLlamaModelInferPolicy", "NoPaddingBaiChuanModelInferPolicy", "model_polic_map"]
