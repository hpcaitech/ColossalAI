import pytest
import torch
import transformers
from hf_tracer_utils import trace_model_and_compare_output

from colossalai.fx import symbolic_trace

try:
    import diffusers
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

BATCH_SIZE = 2
SEQ_LENGTH = 5
HEIGHT = 224
WIDTH = 224
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 8, WIDTH // 8)
TIME_STEP = 2


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers has not been installed")
def test_vae():
    MODEL_LIST = [
        diffusers.AutoencoderKL,
        diffusers.VQModel,
    ]

    for model_cls in MODEL_LIST:
        model = model_cls()
        sample = torch.zeros(LATENTS_SHAPE)

        gm = symbolic_trace(model)

        model.eval()
        gm.eval()

        with torch.no_grad():
            fx_out = gm(sample)
            non_fx_out = model(sample)
        assert torch.allclose(
            fx_out['sample'],
            non_fx_out['sample']), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


def test_clip():
    MODEL_LIST = [
        transformers.CLIPModel,
        transformers.CLIPTextModel,
        transformers.CLIPVisionModel,
    ]

    CONFIG_LIST = [
        transformers.CLIPConfig,
        transformers.CLIPTextConfig,
        transformers.CLIPVisionConfig,
    ]

    def data_gen():
        if isinstance(model, transformers.CLIPModel):
            input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
            attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
            position_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
            pixel_values = torch.zeros((BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
            kwargs = dict(input_ids=input_ids,
                          attention_mask=attention_mask,
                          position_ids=position_ids,
                          pixel_values=pixel_values)
        elif isinstance(model, transformers.CLIPTextModel):
            input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
            attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        elif isinstance(model, transformers.CLIPVisionModel):
            pixel_values = torch.zeros((BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
            kwargs = dict(pixel_values=pixel_values)
        return kwargs

    for model_cls, config in zip(MODEL_LIST, CONFIG_LIST):
        model = model_cls(config=config())
        trace_model_and_compare_output(model, data_gen)


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers has not been installed")
@pytest.mark.skip(reason='cannot pass the test yet')
def test_unet():
    MODEL_LIST = [
        diffusers.UNet2DModel,
        diffusers.UNet2DConditionModel,
    ]

    for model_cls in MODEL_LIST:
        model = model_cls()
        sample = torch.zeros(LATENTS_SHAPE)

        gm = symbolic_trace(model)

        model.eval()
        gm.eval()

        with torch.no_grad():
            fx_out = gm(sample, TIME_STEP)
            non_fx_out = model(sample, TIME_STEP)
        assert torch.allclose(
            fx_out['sample'],
            non_fx_out['sample']), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


if __name__ == "__main__":
    test_vae()
    test_clip()

    # skip because of failure
    # test_unet()
