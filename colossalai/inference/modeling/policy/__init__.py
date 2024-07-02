from .glide_llama import GlideLlamaModelPolicy
from .nopadding_baichuan import NoPaddingBaichuanModelInferPolicy
from .nopadding_llama import NoPaddingLlamaModelInferPolicy
from .pixart_alpha import PixArtAlphaInferPolicy
from .stablediffusion3 import StableDiffusion3InferPolicy

model_policy_map = {
    "nopadding_llama": NoPaddingLlamaModelInferPolicy,
    "nopadding_baichuan": NoPaddingBaichuanModelInferPolicy,
    "glide_llama": GlideLlamaModelPolicy,
    "DiffusionPipe": StableDiffusion3InferPolicy,
}

__all__ = [
    "NoPaddingLlamaModelInferPolicy",
    "NoPaddingBaichuanModelInferPolicy",
    "GlideLlamaModelPolicy",
    "StableDiffusion3InferPolicy",
    "PixArtAlphaInferPolicy",
    "model_polic_map",
]
