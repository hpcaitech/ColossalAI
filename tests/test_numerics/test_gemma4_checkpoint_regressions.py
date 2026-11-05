from functools import partial

import pytest
import torch

from colossalai.booster.plugin import HybridParallelPlugin
from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.acceptance import CONFIGS, _entry_config, run_acceptance
from tests.kit.numerics.scenarios import resume

ENTRY = "transformers_gemma4_e2b_for_causal_lm"
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")


def test_fp16_entry_recipe_preserves_defaults():
    assert _entry_config("ddp_fp16", NumericsSpec()) is CONFIGS["ddp_fp16"]
    config = _entry_config("ddp_fp16", NumericsSpec(fp16_loss_scale=1024))
    assert config.mixed_precision.torch_amp_kwargs["init_scale"] == 1024
    assert CONFIGS["ddp_fp16"].mixed_precision == "fp16"
    assert _entry_config("ddp_fp32", NumericsSpec(fp16_loss_scale=1024)) is CONFIGS["ddp_fp32"]


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_fp16_entry_recipe_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="positive and finite"):
        _entry_config("ddp_fp16", NumericsSpec(fp16_loss_scale=scale))


@GPU
@pytest.mark.parametrize(
    "name", ["ddp_fp16", "gemini_fp16", "shardformer_bf16", "gemini_bf16", "fsdp_fp32", "fsdp_bf16"]
)
def test_gemma4_checkpoint_resume_and_hf(name):
    from tests.kit.model_zoo import model_zoo

    config = _entry_config(name, model_zoo[ENTRY][4].numerics)
    resume.check(ENTRY, config.plugin_fn, config.mixed_precision).assert_pass()


@GPU
@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize("steps_before", [0, 2])
def test_hybrid_master_checkpoint_before_and_after_updates(shard, steps_before):
    # Before Adam's first update there are no moment states; the master must still be included and removed on load.
    plugin = partial(HybridParallelPlugin, tp_size=1, pp_size=1, precision="bf16")
    resume.check(ENTRY, plugin, shard_optimizer=shard, steps_before=steps_before).assert_pass()


@GPU
@pytest.mark.parametrize("name", ["ddp_fp16", "gemini_fp16"])
def test_gemma4_fp16_finite_updates(name):
    report = run_acceptance(ENTRY, configs=[name], checks=["independent_impl"], steps=3)
    assert report["summary"]["passed"] == 1, report
    assert report["recipe_overrides"] == {"fp16_initial_loss_scale": 1024.0}
