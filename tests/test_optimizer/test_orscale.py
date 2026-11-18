import copy
import json
import math

import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import OrScale, muon_param_groups
from colossalai.nn.optimizer.muon import zeropower_via_newtonschulz
from colossalai.nn.optimizer.orscale import polar_express
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.numerics import assert_aligned, judge, judge_relative_bound
from tests.kit.numerics.scenarios.independent_impl import check_optimizer
from tests.test_optimizer.orscale_reference import OrScaleOptimizer, orthogonalize, polar_express_float64

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
STEPS, LR, ADAMW_LR = 3, 0.02, 2e-3

# name: (shape, std of the initial weights). With these stds the trust ratio of "orscale", about ||W|| / ||Q||, starts
# at 0, near 0.05, 0.8, 1.2 and 3.5: clipped from below (twice), inside [0.5, 1.5] (twice) and clipped from above.
MATRICES = {
    "zero": ((24, 16), 0.0),
    "below": ((16, 24), 0.01),
    "tall": ((64, 32), 0.1),
    "wide": ((32, 64), 0.15),
    "above": ((48, 48), 0.5),
}
OTHERS = {"bias": (64,), "gain": (32,)}  # go to the AdamW branch

# check_optimizer judges the weights, and judge's default tol (1e-4 of their RMS) would let through an error of a few
# percent in a step that changes them by a fraction of a percent.
WEIGHT_TOL = 1e-7

CONFIGS = {
    "orscale": dict(variant="orscale", weight_decay=0.1),
    "orscale-no-decay": dict(variant="orscale"),
    "orscale-lm": dict(variant="orscale_lm", weight_decay=0.1, r_min=0.1, r_max=5.0),
    "orscale-lm-no-decay": dict(variant="orscale_lm", r_min=0.1, r_max=5.0),
    "orscale-lm-given-c": dict(variant="orscale_lm", weight_decay=0.1, r_min=0.1, r_max=5.0, c_denom=0.05),
}


def make_params(device="cpu", others=False):
    gen = torch.Generator().manual_seed(0)
    params = {n: std * torch.randn(shape, generator=gen, dtype=torch.float64) for n, (shape, std) in MATRICES.items()}
    if others:
        params.update({n: torch.randn(shape, generator=gen, dtype=torch.float64) for n, shape in OTHERS.items()})
    return {n: p.to(device) for n, p in params.items()}


def ours(cls=OrScale, **kwargs):
    return lambda params: cls(params, **{"lr": LR, **kwargs})


def official(**kwargs):
    return lambda params: OrScaleOptimizer(params, lr=LR, **kwargs)


def check(test_fn, config, device="cpu"):
    return check_optimizer(test_fn, official(**config), make_params(device), torch.float32, steps=STEPS, tol=WEIGHT_TOL)


# ---------------------------------------------------------------------------
# Polar Express
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_polar_express_matches_official(device, shape):
    grad = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(device)
    truth = polar_express_float64(grad)
    ref = orthogonalize(grad.float())
    assert_aligned(polar_express(grad.float()), ref, truth)
    # Muon's Newton-Schulz coefficients leave the singular values in about [0.5, 1.5]: not the same function
    assert not judge(zeropower_via_newtonschulz(grad.float()), ref, truth).passed


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_polar_express_approximates_the_polar_factor(shape):
    # independent of the official code: guards the coefficient table itself
    grad = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    u, _, vh = torch.linalg.svd(grad, full_matrices=False)
    polar = u @ vh
    error = (polar_express(grad.float()).double() - polar).norm() / polar.norm()
    assert error < 0.1, f"relative error {error:.3f} against U V^T"


# ---------------------------------------------------------------------------
# the update rule against the official implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_orscale_matches_official(device, config):
    check(ours(**config), config, device).assert_pass()


@pytest.mark.parametrize(
    "config, wrong",
    [
        ("orscale", {"lr": LR * 1.01}),
        ("orscale", {"momentum": 0.94}),
        ("orscale", {"weight_decay": 0.09}),
        ("orscale", {"r_min": 0.6}),
        ("orscale", {"r_max": 1.4}),
        ("orscale", {"eps": 1e-3}),
        ("orscale", {"ns_steps": 4}),
        ("orscale", {"variant": "orscale_lm"}),
        ("orscale-lm", {"weight_decay": 0.09}),
        ("orscale-lm", {"c_denom": 0.05}),
        ("orscale-lm-given-c", {"c_denom": 0.051}),
    ],
)
def test_alignment_check_catches_wrong_hyperparameters(config, wrong):
    config = CONFIGS[config]
    assert not check(ours(**{**config, **wrong}), config).passed


class DecoupledDecay(OrScale):
    """Weight decay outside the trust ratio, as Muon applies it."""

    @staticmethod
    def _scaled_step(p, update, ratio, scale, group):
        lr = group["lr"]
        p.mul_(1 - lr * group["weight_decay"])
        p.add_(update.to(p.dtype), alpha=-lr * ratio * scale)


class AveragedMomentum(OrScale):
    """Muon's exponential moving average: the same direction, but its bf16 cast before Polar Express rounds differently
    from the accumulated momentum's, which already exceeds the float32 baseline error."""

    def _momentum_update(self, p, group):
        grad, momentum = p.grad.float(), group["momentum"]
        buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(grad))
        buf.lerp_(grad, 1 - momentum)
        return grad.lerp(buf, momentum)


@pytest.mark.parametrize("cls", [DecoupledDecay, AveragedMomentum])
@pytest.mark.parametrize("config", ["orscale", "orscale-lm"])
def test_alignment_check_catches_other_update_rules(cls, config):
    config = CONFIGS[config]
    assert not check(ours(cls, **config), config).passed


def grouped(params, cls, **kwargs):
    params = list(params)
    matrices, others = [p for p in params if p.ndim == 2], [p for p in params if p.ndim != 2]
    return cls(matrices, others, **kwargs)


class OfficialPair:
    """The official setup: OrScale for the matrices and torch's AdamW for everything else, stepped together."""

    def __init__(self, matrices, others, **kwargs):
        self.optimizers = [
            OrScaleOptimizer(matrices, lr=LR, **kwargs),
            torch.optim.AdamW(others, lr=ADAMW_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False),
        ]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)


def one_optimizer(matrices, others, **kwargs):
    groups = [
        {"params": matrices, "use_muon": True},
        {"params": others, "use_muon": False, "lr": ADAMW_LR, "weight_decay": 0.01},
    ]
    return OrScale(groups, lr=LR, betas=(0.9, 0.999), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("config", ["orscale", "orscale-lm"])
def test_orscale_with_adamw_matches_official_pair(device, config):
    config = CONFIGS[config]
    check_optimizer(
        lambda params: grouped(params, one_optimizer, **config),
        lambda params: grouped(params, OfficialPair, **config),
        make_params(device, others=True),
        torch.float32,
        steps=STEPS,
        tol=WEIGHT_TOL,
    ).assert_pass()


def test_orscale_lm_first_step_has_trust_ratio_one():
    # the calibration makes the first step of every layer Moonlight's, whatever the scale of its weights
    params = {n: nn.Parameter(p.float()) for n, p in make_params().items() if n != "zero"}
    start = {n: p.detach().clone() for n, p in params.items()}
    optimizer = OrScale(params.values(), lr=LR, variant="orscale_lm", r_min=0.1, r_max=5.0)
    for p in params.values():
        p.grad = torch.randn_like(p)
    optimizer.step()
    for name, p in params.items():
        # momentum starts at 0, so the Nesterov direction is (1 + momentum) * grad
        direction = polar_express(p.grad * 1.95).float()
        expected = start[name] - LR * 0.2 * math.sqrt(max(p.shape)) * direction
        torch.testing.assert_close(p.detach(), expected, rtol=1e-5, atol=1e-7)
        assert isinstance(optimizer.state[p]["c_denom"], float)


# ---------------------------------------------------------------------------
# parameter grouping, state and error handling
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 16)
        self.proj = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 50, bias=False)

    def get_output_embeddings(self):
        return self.lm_head


def test_orscale_takes_muon_param_groups():
    model = TinyLM()
    optimizer = OrScale(muon_param_groups(model), variant="orscale_lm")
    groups = {g["use_muon"]: g for g in optimizer.param_groups}
    assert [id(p) for p in groups[True]["params"]] == [id(model.proj.weight)]
    assert groups[True]["variant"] == "orscale_lm" and "variant" not in groups[False]
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    optimizer.step()
    assert all(not torch.equal(p, start[n]) for n, p in model.named_parameters())


def test_orscale_state_dict_round_trip():
    torch.manual_seed(0)
    model = TinyLM()
    kwargs = dict(lr=LR, weight_decay=0.1, variant="orscale_lm", r_min=0.1, r_max=5.0)
    optimizer = OrScale(muon_param_groups(model), **kwargs)
    grads = [[torch.randn_like(p) for p in model.parameters()] for _ in range(3)]

    def step(module, opt, step_grads):
        for p, g in zip(module.parameters(), step_grads):
            p.grad = g.clone()
        opt.step()

    step(model, optimizer, grads[0])
    step(model, optimizer, grads[1])
    state = optimizer.state_dict()
    # Hyperparameters and the calibrated constants stay plain data so JSON-based checkpoint formats can store them.
    json.dumps([{k: v for k, v in g.items() if k != "params"} for g in state["param_groups"]])
    assert isinstance(optimizer.state[model.proj.weight]["c_denom"], float)

    resumed_model = copy.deepcopy(model)
    resumed = OrScale(muon_param_groups(resumed_model), **kwargs)
    resumed.load_state_dict(copy.deepcopy(state))
    step(model, optimizer, grads[2])
    step(resumed_model, resumed, grads[2])
    for p, q in zip(model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(q, p, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"variant": "muscale"}, "variant"),
        ({"momentum": 1.0}, "momentum"),
        ({"r_min": 2.0}, "r_min"),
        ({"ns_steps": 6}, "ns_steps"),
        ({"c_denom": 0.0}, "c_denom"),
    ],
)
def test_orscale_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OrScale([nn.Parameter(torch.randn(4, 4))], **kwargs)
    optimizer = OrScale([nn.Parameter(torch.randn(4, 4))])
    with pytest.raises(ValueError, match=match):
        optimizer.add_param_group({"params": [nn.Parameter(torch.randn(4, 4))], **kwargs})
    assert len(optimizer.param_groups) == 1


def test_orscale_rejects_non_matrix_params():
    with pytest.raises(ValueError, match="OrScale only supports 2D"):
        OrScale(nn.Linear(4, 4).parameters())


# ---------------------------------------------------------------------------
# DDP: two ranks on half batches must match one device on the full batch
# ---------------------------------------------------------------------------

DIM, GLOBAL_BATCH = 32, 8


def make_mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(DIM, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, DIM))


def train(model, batches, variant, booster=None, rank=0, world_size=1):
    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to(device)
    start = {n: p.detach().double().clone() for n, p in model.named_parameters()}
    groups = muon_param_groups(model)
    groups[1]["lr"] = ADAMW_LR
    opt = OrScale(groups, lr=LR, weight_decay=0.1, variant=variant)
    if booster is not None:
        model, opt, *_ = booster.boost(model, opt)
    local = GLOBAL_BATCH // world_size
    for batch in batches:
        x = batch[rank * local : (rank + 1) * local].to(device)
        loss = model(x).pow(2).mean()
        if booster is not None:
            booster.backward(loss, opt)
        else:
            loss.backward()
        opt.step()
        opt.zero_grad()
    module = model.unwrap() if hasattr(model, "unwrap") else model
    return {n: (p.detach().double() - start[n]).cpu() for n, p in module.named_parameters()}


def run_ddp(rank, world_size, port, variant):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    batches = torch.randn(STEPS, GLOBAL_BATCH, DIM, generator=torch.Generator().manual_seed(2))
    booster = Booster(plugin=TorchDDPPlugin())

    test = train(make_mlp(), batches, variant, booster, rank, world_size)
    ref = train(make_mlp(), batches, variant)
    # Same code at the same precision, so there is no baseline error to measure: the tool's relative bound applies,
    # at the bf16 of the orthogonalization, which turns float32 rounding of the averaged gradient into bf16 rounding.
    judge_relative_bound(test, ref, torch.bfloat16).assert_pass()
    # Each rank stepping on its own half batch (no gradient all-reduce) must be caught.
    unsynced = train(make_mlp(), batches, variant, rank=rank, world_size=world_size)
    assert not judge_relative_bound(unsynced, ref, torch.bfloat16).passed


@pytest.mark.dist
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("variant", ["orscale", "orscale_lm"])
@rerun_if_address_is_in_use()
def test_orscale_ddp(variant):
    spawn(run_ddp, nprocs=2, variant=variant)


if __name__ == "__main__":
    test_orscale_ddp("orscale")
