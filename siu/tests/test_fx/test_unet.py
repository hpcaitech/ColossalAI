import diffusers
import pytest
import torch

from siu.fx import symbolic_trace

BATCH_SIZE = 2
SEQ_LENGTH = 5
HEIGHT = 224
WIDTH = 224
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 7, WIDTH // 7)
TIME_STEP = 3

VAE_ARG_LIST = [
    (diffusers.AutoencoderKL, LATENTS_SHAPE, {}),
    (diffusers.VQModel, LATENTS_SHAPE, {}),
]


@pytest.mark.parametrize('m, shape, kwargs', VAE_ARG_LIST)
def test_vae(m, shape, kwargs):

    model = m()
    sample = torch.zeros(shape)

    gm = symbolic_trace(model, meta_args={'sample': sample}, concrete_args=kwargs)

    model.eval()
    gm.eval()

    with torch.no_grad():
        fx_out = gm(sample)
        non_fx_out = model(sample)
    assert torch.allclose(
        fx_out['sample'],
        non_fx_out['sample']), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


UNET_ARG_LIST = [
    (diffusers.UNet2DModel, LATENTS_SHAPE, TIME_STEP),
]


@pytest.mark.parametrize('m, shape, timestep', UNET_ARG_LIST)
def test_unet(m, shape, timestep):
    model = m()
    sample = torch.zeros(shape)

    gm = symbolic_trace(model, meta_args={'sample': sample}, concrete_args=dict(timestep=timestep))

    model.eval()
    gm.eval()

    with torch.no_grad():
        fx_out = gm(sample, timestep)
        non_fx_out = model(sample, timestep)
    assert torch.allclose(
        fx_out['sample'],
        non_fx_out['sample']), f'{model.__class__.__name__} has inconsistent outputs, {fx_out} vs {non_fx_out}'


if __name__ == "__main__":
    test_vae(*VAE_ARG_LIST[1])
    test_unet(*UNET_ARG_LIST[0])
