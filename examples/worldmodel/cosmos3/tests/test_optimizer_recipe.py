"""The three places where this example's optimizer used to leave the recipe.

A constant learning rate, no EMA and one extra parameter group are all things a training run does not
complain about: the loss still falls, the step still takes the same time, and the difference shows up only
as a slow divergence from the reference's curve. Each one is pinned here against the reference's own
numbers rather than against what this file happens to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="needs torch")
train_sft = pytest.importorskip("train_sft", reason="needs torch, colossalai and the example's own modules")


# --- the schedule ------------------------------------------------------------------------------------------
def test_the_first_step_runs_at_the_rate_the_recipe_asks_for():
    """f_max is 0.4 and the warm-up is empty, so step zero is 8e-05 and not the configured 2e-04."""
    assert train_sft.lr_multiplier(0) == pytest.approx(0.4)
    assert train_sft.LR * train_sft.lr_multiplier(0) == pytest.approx(8.0e-05)


def test_the_multiplier_falls_linearly_across_the_cycle():
    cycle = train_sft.LR_CYCLE_LENGTH
    assert train_sft.lr_multiplier(cycle // 2) == pytest.approx(0.2)
    assert train_sft.lr_multiplier(cycle // 4) == pytest.approx(0.3)
    # Equal steps apart, equal drops: that is what "linear" has to mean here.
    first = train_sft.lr_multiplier(10) - train_sft.lr_multiplier(11)
    later = train_sft.lr_multiplier(5000) - train_sft.lr_multiplier(5001)
    assert first == pytest.approx(later)


def test_the_multiplier_never_goes_negative_past_the_cycle():
    """The reference raises here instead; a negative rate would be worse than either."""
    cycle = train_sft.LR_CYCLE_LENGTH
    assert train_sft.lr_multiplier(cycle) == pytest.approx(0.0)
    assert train_sft.lr_multiplier(cycle * 3) == pytest.approx(0.0)


class _Groups:
    """Just enough optimizer for set_learning_rate: the booster's wrapper exposes param_groups too."""

    def __init__(self, lrs):
        self.param_groups = [{"lr": lr} for lr in lrs]


def test_the_rate_is_written_on_every_group_and_the_5x_ratio_survives():
    opt = _Groups([train_sft.LR, train_sft.LR * 5.0])
    train_sft.set_learning_rate(opt, 0)
    assert opt.param_groups[0]["lr"] == pytest.approx(8.0e-05)
    assert opt.param_groups[1]["lr"] == pytest.approx(4.0e-04)
    # Called again for a later step it must scale the *original* rate, not the one it last wrote.
    train_sft.set_learning_rate(opt, train_sft.LR_CYCLE_LENGTH // 2)
    assert opt.param_groups[0]["lr"] == pytest.approx(train_sft.LR * 0.2)
    assert opt.param_groups[1]["lr"] == pytest.approx(train_sft.LR * 5.0 * 0.2)


# --- the EMA -----------------------------------------------------------------------------------------------
class _Model:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class _Optimizer:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_working_to_master_map(self):
        return self._mapping


def _ema(values):
    working = [torch.nn.Parameter(torch.tensor(v)) for v in values]
    masters = {id(p): p.detach().float().clone() for p in working}
    return train_sft.PowerEMA(_Model(working), _Optimizer(masters)), masters, working


def test_the_decay_exponent_is_the_root_the_reference_computes():
    """s = 0.1 is the EDM2 rate, not a decay; the exponent is a root of a cubic in it."""
    ema, _, _ = _ema([[1.0, 2.0]])
    assert ema.exponent == pytest.approx(6.937203937601809)


def test_the_first_update_copies_rather_than_averages():
    """iteration 0 means beta 0, which is the reference's warm start."""
    ema, masters, working = _ema([[1.0, 2.0]])
    assert ema.beta(0) == 0.0
    for t in masters.values():
        t.mul_(0.0).add_(7.0)
    ema.update(0)
    assert torch.equal(ema.shadow[0], torch.tensor([7.0, 7.0]))


def test_a_later_update_mixes_with_the_reference_coefficient():
    ema, masters, working = _ema([[0.0, 0.0]])
    beta = ema.beta(1)
    assert beta == pytest.approx((1.0 - 1.0 / 2.0) ** (ema.exponent + 1.0))
    for t in masters.values():
        t.add_(1.0)
    ema.update(1)
    assert ema.shadow[0][0].item() == pytest.approx(1.0 - beta)


def test_it_averages_the_fp32_masters_and_not_the_bf16_working_copy():
    """Reading the working parameters would feed a rounded source into an fp32 average."""
    working = [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))]
    masters = {id(working[0]): torch.tensor([1.0 + 2.0**-20], dtype=torch.float32)}
    ema = train_sft.PowerEMA(_Model(working), _Optimizer(masters))
    assert ema.shadow[0].dtype is torch.float32
    assert ema.shadow[0].item() != pytest.approx(1.0, abs=0.0)
    assert ema.shadow[0].item() == pytest.approx(1.0 + 2.0**-20)


def test_it_falls_back_to_the_parameters_when_there_is_no_master_copy_and_says_so():
    """At zero_stage 0 there is no fp32 master to read, and the parameters are the only source there is.

    Averaging them is correct there and wrong at zero_stage 2, where the masters are fp32 and the working
    copy is bf16, so the source is reported rather than assumed -- a silent substitution would round the
    input of an average whose whole point is that it is not rounded.
    """
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    ema = train_sft.PowerEMA(_Model([parameter]), _Optimizer({}))

    assert ema.sources == [parameter]
    assert "no ZeRO master copy" in ema.source_kind


def test_it_refuses_to_average_nothing():
    """A model with nothing trainable leaves the EMA no source at all, which is a configuration error."""
    with pytest.raises(RuntimeError, match="no trainable parameter"):
        train_sft.PowerEMA(_Model([]), _Optimizer({}))


# --- the key set -------------------------------------------------------------------------------------------
def test_the_understanding_towers_key_norm_is_not_trained():
    """The reference's keys_to_select matches by substring, and no entry of it matches this name.

    Measured on one step: the reference's gradient dump holds 294 tensors and this example's held 322, and
    the 28 extra ones were exactly these.
    """
    assert "k_norm_und_for_gen" not in train_sft.TRAINABLE_KEYS
    assert not any(k in "layers.0.self_attn.k_norm_und_for_gen.weight" for k in train_sft.TRAINABLE_KEYS)
    # The generation tower's own key norm stays trainable; only the understanding one is excluded.
    assert any(k in "layers.0.self_attn.k_norm_moe_gen.weight" for k in train_sft.TRAINABLE_KEYS)
