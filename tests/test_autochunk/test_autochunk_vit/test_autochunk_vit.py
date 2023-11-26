from typing import List, Tuple

import pytest
import torch

try:
    from timm.models.vision_transformer import vit_large_patch16_384 as vit

    MODELS = [vit]
    HAS_REPO = True
except:
    MODELS = []
    HAS_REPO = False

from test_autochunk_vit_utils import run_test

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.testing import clear_cache_before_run, parameterize, spawn


def get_data() -> Tuple[List, List]:
    data = torch.rand(1, 3, 384, 384)
    meta_args = {"x": data}
    return data, meta_args


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@clear_cache_before_run()
@parameterize("model", MODELS)
@parameterize("max_memory", [None, 32, 40])
def test_evoformer_block(model, max_memory):
    spawn(
        run_test,
        1,
        max_memory=max_memory,
        model=model,
        data=get_data(),
    )


if __name__ == "__main__":
    run_test(
        rank=0,
        data=get_data(),
        max_memory=None,
        model=vit,
        print_code=False,
        print_mem=False,
        print_est_mem=False,
        print_progress=False,
    )
