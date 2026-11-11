import importlib
import importlib.util
import os
import warnings

from . import custom, diffusers

# Text-only CPU environments need not install the optional vision/audio model families.
# Check package availability before importing; errors in installed packages must still surface.
for _family in ("timm", "torchaudio", "torchvision"):
    if importlib.util.find_spec(_family) is not None:
        globals()[_family] = importlib.import_module(f".{_family}", __name__)
    else:
        warnings.warn(f"Optional model-zoo family {_family!r} is not registered: package is not installed.")

from . import transformers
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
    "transformers_llama_for_causal_lm",
    "transformers_vit_for_masked_image_modeling",
    "transformers_mistral_for_causal_lm",
]

IS_FAST_TEST = os.environ.get("FAST_TEST", "0") == "1"


__all__ = ["model_zoo", "run_fwd", "run_fwd_bwd", "COMMON_MODELS", "IS_FAST_TEST"]
