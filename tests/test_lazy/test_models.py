import pytest
from lazy_init_utils import SUPPORT_LAZY, check_lazy_init

from tests.kit.model_zoo import COMMON_MODELS, IS_FAST_TEST, model_zoo


@pytest.mark.skipif(not SUPPORT_LAZY, reason="requires torch >= 1.12.0")
@pytest.mark.parametrize(
    "subset",
    (
        [COMMON_MODELS]
        if IS_FAST_TEST
        else ["torchvision", "diffusers", "timm", "transformers", "torchaudio", "deepfm", "dlrm"]
    ),
)
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_models_lazy_init(subset, default_device):
    sub_model_zoo = model_zoo.get_sub_registry(subset, allow_empty=True)
    for name, entry in sub_model_zoo.items():
        # TODO(ver217): lazy init does not support weight norm, skip these models
        if name in (
            "torchaudio_wav2vec2_base",
            "torchaudio_hubert_base",
            "timm_beit",
            "timm_vision_transformer",
            "timm_deit",
            "timm_beitv2",
            "timm_deit3",
            "timm_convit",
            "timm_tnt_b_patch16_224",
        ) or name.startswith(("transformers_vit", "transformers_blip2", "transformers_whisper")):
            continue
        check_lazy_init(entry, verbose=True, default_device=default_device)


if __name__ == "__main__":
    test_models_lazy_init("transformers", "cpu")
