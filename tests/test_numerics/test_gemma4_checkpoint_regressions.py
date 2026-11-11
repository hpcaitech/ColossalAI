from functools import partial
from types import SimpleNamespace

import pytest
import torch

from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import (
    HybridParallelCheckpointIO,
    _optimizer_checkpoint_states,
)
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.acceptance import CONFIGS, _entry_config, run_acceptance
from tests.kit.numerics.scenarios import resume

ENTRY = "transformers_gemma4_e2b_for_causal_lm"
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU")


@pytest.mark.parametrize("legacy", [False, True])
def test_master_state_metadata_is_private_and_backward_compatible(legacy):
    working = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    master = torch.ones(3) + 0.0001
    optim = torch.optim.AdamW([master])
    wrapper = SimpleNamespace(
        optim=optim,
        param_info={"param2shape": {id(working): working.shape}},
        get_master_to_working_map=lambda: {id(master): working},
    )
    state = dict(next(_optimizer_checkpoint_states(wrapper))[1])
    original = master.clone()
    if legacy:
        state.pop("_colossalai_master_param")
        state["exp_avg"] = torch.ones(3)
    else:
        state["_colossalai_master_param"] = original.clone()
    master.zero_()
    io = HybridParallelCheckpointIO.__new__(HybridParallelCheckpointIO)
    io.shard_from_complete_optimizer_state = lambda state, **kwargs: state
    io.load_states_into_optimizer(wrapper, {0: state}, {0: master})
    assert "_colossalai_master_param" not in optim.state.get(master, {})
    if legacy:
        assert torch.equal(master, torch.zeros(3))
        assert torch.equal(optim.state[master]["exp_avg"], torch.ones(3))
    else:
        assert torch.equal(master, original)
        assert master not in optim.state


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
@rerun_if_address_is_in_use()
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


def _amp_checkpoint_worker(rank, world_size, port, directory, shard):
    import colossalai
    from colossalai.booster import Booster
    from colossalai.booster.mixed_precision.fp16_torch import FP16TorchMixedPrecision
    from colossalai.booster.plugin import TorchDDPPlugin

    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)

    def build(scale, growth_interval):
        model = torch.nn.Linear(3, 2).cuda()
        optimizer = torch.optim.AdamW(model.parameters())
        booster = Booster(
            plugin=TorchDDPPlugin(),
            mixed_precision=FP16TorchMixedPrecision(init_scale=scale, growth_interval=growth_interval),
        )
        model, optimizer, *_ = booster.boost(model, optimizer)
        return booster, model, optimizer

    booster, model, optimizer = build(1024, 2)
    loss = model(torch.ones(1, 3, device="cuda")).float().sum()
    booster.backward(loss, optimizer)
    optimizer.step()
    optimizer.zero_grad()
    state = optimizer.scaler.state_dict()
    assert state["_growth_tracker"] == 1
    booster.save_optimizer(optimizer, directory, shard=shard)
    booster, model, loaded = build(8, 17)
    booster.load_optimizer(loaded, directory)
    assert loaded.scaler.state_dict() == state
    assert "_colossalai_grad_scaler" not in loaded.param_groups[0]
    loss = model(torch.ones(1, 3, device="cuda")).float().sum()
    booster.backward(loss, loaded)
    loaded.step()
    assert loaded.scaler.get_scale() == 2048


@GPU
@pytest.mark.parametrize("shard", [False, True])
def test_ddp_amp_scaler_checkpoint(shard, tmp_path):
    from colossalai.testing import spawn

    path = str(tmp_path / ("optimizer" if shard else "optimizer.bin"))
    spawn(_amp_checkpoint_worker, 1, directory=path, shard=shard)
