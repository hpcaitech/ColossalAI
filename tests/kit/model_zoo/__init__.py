import os

from . import custom, diffusers, timm, torchaudio, torchvision, transformers
from .executor import run_fwd, run_fwd_bwd
from .registry import model_zoo

# We pick a subset of models for fast testing in order to reduce the total testing time
COMMON_MODELS = [
    "custom_hanging_param_model",
    "custom_nested_model",
    "custom_repeated_computed_layers",
    "custom_simple_net",
    "diffusers_clip_text_model",
    "diffusers_auto_encoder_kl",
    "diffusers_unet2d_model",
    "timm_densenet",
    "timm_resnet",
    "timm_swin_transformer",
    "torchaudio_wav2vec2_base",
    "torchaudio_conformer",
    "transformers_bert_for_masked_lm",
    "transformers_bloom_for_causal_lm",
    "transformers_falcon_for_causal_lm",
    "transformers_chatglm_for_conditional_generation",
    "transformers_llama_for_casual_lm",
    "transformers_vit_for_masked_image_modeling",
    "transformers_mistral_for_casual_lm",
]

IS_FAST_TEST = os.environ.get("FAST_TEST", "0") == "1"


__all__ = ["model_zoo", "run_fwd", "run_fwd_bwd", "COMMON_MODELS", "IS_FAST_TEST"]
