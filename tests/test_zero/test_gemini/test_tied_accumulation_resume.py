"""Shared Gemini regression from the ac09ab3b handoff: tied weights, accumulation and optimizer resume."""

import pytest
import torch
from torch import nn


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(8, 8, bias=False)
        self.right = nn.Linear(8, 8, bias=False)
        self.right.weight = self.left.weight

    def forward(self, x):
        return self.right(self.left(x).tanh())


def _data():
    return {"x": torch.randn(4, 8)}


def _outputs(x):
    return {"output": x}


def _gemini(keep_gathered):
    from colossalai.booster.plugin import GeminiPlugin

    return GeminiPlugin(
        precision="bf16",
        enable_gradient_accumulation=True,
        chunk_config_dict={size: {"chunk_size": 512, "keep_gathered": keep_gathered} for size in (1, 2)},
        chunk_init_device=torch.device("cuda"),
    )


def _entry():
    from tests.kit.model_zoo.registry import ModelAttribute
    from tests.kit.numerics import NumericsSpec

    spec = NumericsSpec(tied_params=(("left.weight", "right.weight"),))
    return TiedModel, _data, _outputs, None, ModelAttribute(numerics=spec)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("keep_gathered", [False, True])
def test_gemini_accumulation_and_tied_resume(keep_gathered):
    from functools import partial

    from tests.kit.numerics.scenarios import data_parallel, resume

    factory = partial(_gemini, keep_gathered)
    # Two ranks with two micro-batches each must match one rank with four.
    data_parallel.check(_entry(), factory, world_size=2, micro_batches=4).assert_pass()
    # Exercise both gathered and sharded optimizer state, including master weights.
    resume.check(_entry(), factory, world_size=2, shard_optimizer=True).assert_pass()
