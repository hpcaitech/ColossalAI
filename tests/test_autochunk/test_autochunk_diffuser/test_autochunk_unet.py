from typing import List, Tuple

import pytest
import torch

try:
    import diffusers

    MODELS = [diffusers.UNet2DModel]
    HAS_REPO = True
    from packaging import version

    SKIP_UNET_TEST = version.parse(diffusers.__version__) > version.parse("0.10.2")
except:
    MODELS = []
    HAS_REPO = False
    SKIP_UNET_TEST = False

from test_autochunk_diffuser_utils import run_test

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.testing import clear_cache_before_run, parameterize, spawn

BATCH_SIZE = 1
HEIGHT = 448
WIDTH = 448
IN_CHANNELS = 3
LATENTS_SHAPE = (BATCH_SIZE, IN_CHANNELS, HEIGHT // 7, WIDTH // 7)


def get_data(shape: tuple) -> Tuple[List, List]:
    sample = torch.randn(shape)
    meta_args = [
        ("sample", sample),
    ]
    concrete_args = [("timestep", 50)]
    return meta_args, concrete_args


@pytest.mark.skipif(
    SKIP_UNET_TEST,
    reason="diffusers version > 0.10.2",
)
@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@clear_cache_before_run()
@parameterize("model", MODELS)
@parameterize("shape", [LATENTS_SHAPE])
@parameterize("max_memory", [None, 150, 300])
def test_evoformer_block(model, shape, max_memory):
    spawn(
        run_test,
        1,
        max_memory=max_memory,
        model=model,
        data=get_data(shape),
    )


if __name__ == "__main__":
    test_evoformer_block()
