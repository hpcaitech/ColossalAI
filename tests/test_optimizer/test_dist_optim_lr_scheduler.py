"""An LR scheduler built before ``booster.boost`` keeps driving the optimizer that steps.

LowLevelZero, HybridParallel and MoeHybridParallel replace Lamb, CAME, Adafactor and GaLoreAdamW8bit with their
distributed versions (``cast_to_distributed``): a new optimizer on the same ``param_groups`` dicts. A scheduler built on
the original one writes its learning rates to those dicts, which works until a checkpoint is loaded: loading gives the
new optimizer new dicts, and the scheduler then updates an optimizer nobody steps.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin, MoeHybridParallelPlugin
from colossalai.logging import disable_existing_loggers
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import CAME, DistributedLamb, Lamb, cast_to_distributed, optim2DistOptim
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.scenarios import resume

LR = 0.02


def requires_gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"needs {n} GPUs")


def linear(optimizer):
    return LinearLR(optimizer, start_factor=0.25, total_iters=4)


def sequential(optimizer):
    # steps its parts, each of which holds the optimizer too
    parts = [ConstantLR(optimizer, factor=0.5, total_iters=2), LinearLR(optimizer, start_factor=0.25, total_iters=4)]
    return SequentialLR(optimizer, parts, milestones=[2])


def cosine_warmup(optimizer):
    # ColossalAI's two-stage schedulers hand over to `after_scheduler` once the warmup is done
    return CosineAnnealingWarmupLR(optimizer, total_steps=8, warmup_steps=2)


def scheduled_lrs(optimizer_cls, scheduler_fn, cast):
    """The learning rates of the optimizer that steps over six scheduler steps taken after a checkpoint was loaded."""
    optimizer = original = optimizer_cls([nn.Parameter(torch.zeros(4, 4))], lr=LR)
    scheduler = scheduler_fn(optimizer)
    if cast:
        optimizer = cast_to_distributed(original, scheduler)
        assert type(optimizer) is optim2DistOptim[optimizer_cls]
    # as loading a checkpoint does (GeneralCheckpointIO.load_unsharded_optimizer): new param_group dicts
    optimizer.load_state_dict(optimizer.state_dict())
    lrs = []
    for _ in range(6):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


# Of the replaced optimizers, Adafactor always computes its own step size (lr=None), which no scheduler can drive, and
# GaLoreAdamW8bit cannot be built with a bitsandbytes whose Optimizer2State no longer takes `percentile_clipping`
@pytest.mark.parametrize("scheduler_fn", [linear, sequential, cosine_warmup])
@pytest.mark.parametrize("optimizer_cls", [Lamb, CAME])
def test_scheduler_follows_cast_optimizer(optimizer_cls, scheduler_fn):
    expected = scheduled_lrs(optimizer_cls, scheduler_fn, cast=False)
    assert len(set(expected)) > 1  # the schedule moves, so an optimizer left behind shows
    assert scheduled_lrs(optimizer_cls, scheduler_fn, cast=True) == expected


def test_cast_leaves_other_schedulers_alone():
    weight = nn.Parameter(torch.zeros(4, 4))
    # not replaced: nothing to follow
    adamw = torch.optim.AdamW([weight], lr=LR)
    scheduler = linear(adamw)
    assert cast_to_distributed(adamw, scheduler) is adamw and scheduler.optimizer is adamw
    # replaced, but the scheduler drives another optimizer
    lamb = Lamb([weight], lr=LR)
    assert cast_to_distributed(lamb, scheduler) is not lamb and scheduler.optimizer is adamw
    # and without a scheduler, as before
    assert type(cast_to_distributed(Lamb([weight], lr=LR))) is optim2DistOptim[Lamb]


# ---------------------------------------------------------------------------
# On GPUs: every plugin that replaces the optimizer hands the scheduler over
# ---------------------------------------------------------------------------

PLUGINS = {
    "low_level_zero": LowLevelZeroPlugin,
    "hybrid_parallel": partial(HybridParallelPlugin, tp_size=1, pp_size=1),
    "moe_hybrid_parallel": partial(MoeHybridParallelPlugin, tp_size=1, pp_size=1, ep_size=1),
}


def run_boost(rank, world_size, port, plugin_fn):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    # Mixtral has the experts MoeHybridParallelPlugin is made for; the other two plugins take it as well
    config = transformers.MixtralConfig(
        hidden_size=32, intermediate_size=32, num_attention_heads=8, num_hidden_layers=1, vocab_size=128
    )
    model = transformers.MixtralModel(config)
    lamb = Lamb(model.parameters(), lr=LR)
    scheduler = sequential(lamb)
    _, optimizer, _, _, scheduler = Booster(plugin=plugin_fn()).boost(model, lamb, lr_scheduler=scheduler)
    stepping = optimizer.unwrap()
    assert isinstance(stepping, DistributedLamb)
    assert all(s.optimizer is stepping for s in (scheduler, *scheduler._schedulers))
    dist.destroy_process_group()


@pytest.mark.dist
@requires_gpus(1)
@pytest.mark.parametrize("plugin_fn", PLUGINS.values(), ids=PLUGINS.keys())
@rerun_if_address_is_in_use()
def test_boost_moves_scheduler_to_distributed_optimizer(plugin_fn):
    spawn(run_boost, 1, plugin_fn=plugin_fn)


# ---------------------------------------------------------------------------
# LowLevelZeRO on GPUs: training resumed from a checkpoint against training that was never interrupted
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # odd sizes, so that ZeRO pads every parameter it shards
        self.fc1 = nn.Linear(31, 63)
        self.norm = nn.LayerNorm(63)
        self.fc2 = nn.Linear(63, 17)

    def forward(self, x):
        return {"y": self.fc2(self.norm(F.gelu(self.fc1(x))))}


def _data():
    return {"x": torch.randn(8, 31)}


def _identity(output):
    return output


def lamb(params):
    return Lamb(params, lr=LR)


# module-level parts only: the entry and the optimizer function are sent to spawned processes
ENTRY = (MLP, _data, _identity, None, SimpleNamespace(numerics=NumericsSpec()))


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [pytest.param(n, marks=requires_gpus(n)) for n in (1, 2)])
@rerun_if_address_is_in_use()
def test_low_level_zero_resume_is_exact(world_size):
    # the scenario builds a LinearLR on the optimizer before boost, so the learning rate differs at every step.
    # Lamb: its state has the shape of its parameters, which LowLevelZero needs to save it (CAME's factored state has
    # not). fp32: in bf16 no optimizer resumes exactly (tests/kit/numerics/README.md, "Resuming from a checkpoint")
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="fp32")
    resume.check(ENTRY, plugin_fn, world_size=world_size, optimizer_fn=lamb).assert_pass()
