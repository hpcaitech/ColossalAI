import diffusers
import pytest
import torch

from colossalai.fx.tracer.experimental import symbolic_trace

BATCH_SIZE = 2
SEQ_LENGTH = 5
HEIGHT = 224
WIDTH = 224
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 7, WIDTH // 7)
TIME_STEP = 50

ARG_LIST = [
    (diffusers.UNet2DModel, LATENTS_SHAPE, TIME_STEP),
]


@pytest.mark.parametrize('m, shape, timestep', ARG_LIST)
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
    test_unet(*ARG_LIST[0])
