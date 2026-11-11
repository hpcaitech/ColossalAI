import copy
import json

import pytest
import torch
import torch.nn as nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import Muon, muon_param_groups
from colossalai.nn.optimizer.muon import zeropower_via_newtonschulz
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.numerics import assert_aligned, judge, judge_relative_bound
from tests.kit.numerics.scenarios.independent_impl import check_optimizer

requires_torch_muon = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs torch>=2.9")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

MATRIX_SHAPES = [(64, 32), (32, 64), (48, 48)]
OTHER_SHAPES = [(64,), (100, 32)]  # a bias and an embedding: both go to the AdamW branch
STEPS = 3
LR, ADAMW_LR, BETAS, ADAMW_EPS = 0.02, 3e-3, (0.9, 0.95), 1e-8


# ---------------------------------------------------------------------------
# float64 ground truth, transcribed from torch/optim/_muon.py (torch 2.13) without its bfloat16 cast
# ---------------------------------------------------------------------------


def reference_newtonschulz(grad, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.double()
    if grad.size(0) > grad.size(1):
        x = x.T
    x = x / x.norm().clamp(min=1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    if grad.size(0) > grad.size(1):
        x = x.T
    return x


def reference_muon_step(param, grad, buf, lr, weight_decay, momentum, nesterov, adjust_lr_fn):
    buf.lerp_(grad, 1 - momentum)
    update = grad.lerp(buf, momentum) if nesterov else buf
    update = reference_newtonschulz(update)
    rows, cols = param.shape
    if adjust_lr_fn == "match_rms_adamw":
        adjusted_lr = lr * 0.2 * max(rows, cols) ** 0.5
    else:
        adjusted_lr = lr * max(1, rows / cols) ** 0.5
    param.mul_(1 - lr * weight_decay)
    param.add_(update, alpha=-adjusted_lr)


# ---------------------------------------------------------------------------
# single-step and multi-step alignment against torch.optim.Muon / torch.optim.AdamW
# ---------------------------------------------------------------------------


def make_problem():
    gen = torch.Generator().manual_seed(0)
    shapes = MATRIX_SHAPES + OTHER_SHAPES
    init = [torch.randn(s, generator=gen, dtype=torch.float64) for s in shapes]
    grads = [[torch.randn(s, generator=gen, dtype=torch.float64) for s in shapes] for _ in range(STEPS)]
    return init, grads


def run_steps(init, grads, dtype, device, step_fn):
    """Apply ``step_fn`` for every step; return the parameter deltas and the Muon momentum buffers."""
    # copy=True: without it the float64 run would update the shared ``init`` tensors in place.
    params = [nn.Parameter(t.to(device=device, dtype=dtype, copy=True)) for t in init]
    start = [p.detach().clone() for p in params]
    n = len(MATRIX_SHAPES)
    bufs = step_fn(params[:n], params[n:], grads, dtype, device)
    out = {f"delta/{i}": (p.detach().double() - s.double()).cpu() for i, (p, s) in enumerate(zip(params, start))}
    out.update({f"momentum/{i}": b.detach().cpu() for i, b in enumerate(bufs)})
    return out


def colossal_step_fn(**muon_kwargs):
    def step_fn(matrices, others, grads, dtype, device):
        groups = [
            {"params": matrices, "use_muon": True},
            {"params": others, "use_muon": False, "lr": ADAMW_LR},
        ]
        opt = Muon(groups, lr=LR, betas=BETAS, adamw_eps=ADAMW_EPS, **muon_kwargs)
        for step_grads in grads:
            for p, g in zip(matrices + others, step_grads):
                p.grad = g.to(device=device, dtype=dtype, copy=True)
            opt.step()
        return [opt.state[p]["momentum_buffer"] for p in matrices]

    return step_fn


def torch_step_fn(weight_decay, nesterov, adjust_lr_fn, reference_muon):
    def step_fn(matrices, others, grads, dtype, device):
        adamw = torch.optim.AdamW(
            others, lr=ADAMW_LR, betas=BETAS, eps=ADAMW_EPS, weight_decay=weight_decay, foreach=False
        )
        if reference_muon:
            bufs = [torch.zeros_like(p) for p in matrices]
        else:
            muon = torch.optim.Muon(
                matrices, lr=LR, weight_decay=weight_decay, nesterov=nesterov, adjust_lr_fn=adjust_lr_fn
            )
        for step_grads in grads:
            for p, g in zip(matrices + others, step_grads):
                p.grad = g.to(device=device, dtype=dtype, copy=True)
            with torch.no_grad():
                if reference_muon:
                    for p, buf in zip(matrices, bufs):
                        reference_muon_step(p, p.grad, buf, LR, weight_decay, 0.95, nesterov, adjust_lr_fn)
                else:
                    muon.step()
            adamw.step()
        return bufs if reference_muon else [muon.state[p]["momentum_buffer"] for p in matrices]

    return step_fn


@requires_torch_muon
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", MATRIX_SHAPES)
def test_newton_schulz_matches_torch(device, shape):
    from torch.optim._muon import _zeropower_via_newtonschulz

    grad = torch.randn(shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64).to(device)
    truth = reference_newtonschulz(grad)
    ref = _zeropower_via_newtonschulz(grad.float(), (3.4445, -4.7750, 2.0315), 5, 1e-7)
    test = zeropower_via_newtonschulz(grad.float())
    assert_aligned(test, ref, truth)


@requires_torch_muon
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_muon_matches_torch(device, weight_decay, nesterov, adjust_lr_fn):
    init, grads = make_problem()
    hyper = dict(weight_decay=weight_decay, nesterov=nesterov, adjust_lr_fn=adjust_lr_fn)
    truth = run_steps(init, grads, torch.float64, device, torch_step_fn(**hyper, reference_muon=True))
    ref = run_steps(init, grads, torch.float32, device, torch_step_fn(**hyper, reference_muon=False))
    test = run_steps(init, grads, torch.float32, device, colossal_step_fn(**hyper))
    assert_aligned(test, ref, truth)


@requires_torch_muon
@pytest.mark.parametrize(
    "wrong",
    [
        {"adjust_lr_fn": "match_rms_adamw"},
        {"nesterov": False},
        {"momentum": 0.9},
        {"momentum": 0.949},
        {"weight_decay": 0.09},
    ],
)
def test_alignment_check_catches_wrong_hyperparameters(wrong):
    # The baseline-error rule must not let a wrong update through just because bf16 Newton-Schulz is noisy.
    init, grads = make_problem()
    hyper = dict(weight_decay=0.1, nesterov=True, adjust_lr_fn=None)
    truth = run_steps(init, grads, torch.float64, "cpu", torch_step_fn(**hyper, reference_muon=True))
    ref = run_steps(init, grads, torch.float32, "cpu", torch_step_fn(**hyper, reference_muon=False))
    test = run_steps(init, grads, torch.float32, "cpu", colossal_step_fn(**{**hyper, **wrong}))
    assert not judge(test, ref, truth).passed


# check_optimizer judges the weights and judge scales tol by their RMS. At lr=1e-3 the updates are far smaller than
# these unit-variance weights, so the default tol would let lr * 1.05 or momentum 0.9 through its allclose fast path.
WEIGHT_TOL = 1e-7


@requires_torch_muon
@pytest.mark.parametrize("adjust_lr_fn", [None, "match_rms_adamw"])
def test_check_optimizer_against_torch(adjust_lr_fn):
    gen = torch.Generator().manual_seed(3)
    params = {f"matrix{i}": torch.randn(s, generator=gen, dtype=torch.float64) for i, s in enumerate(MATRIX_SHAPES)}
    lr = 1e-3

    def check(**muon_kwargs):
        return check_optimizer(
            lambda ps: Muon(ps, adjust_lr_fn=adjust_lr_fn, **muon_kwargs),
            lambda ps: torch.optim.Muon(ps, lr=lr, adjust_lr_fn=adjust_lr_fn),
            params,
            torch.float32,
            steps=STEPS,
            tol=WEIGHT_TOL,
        )

    check(lr=lr).assert_pass()
    assert not check(lr=lr * 1.05).passed
    assert not check(lr=lr, momentum=0.9).passed


# ---------------------------------------------------------------------------
# parameter grouping and error handling
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    def __init__(self, tie: bool):
        super().__init__()
        self.embed = nn.Embedding(50, 16)
        self.proj = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 50, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def get_output_embeddings(self):
        return self.lm_head


def group_ids(groups):
    return {g["use_muon"]: {id(p) for p in g["params"]} for g in groups}


@pytest.mark.parametrize("tie", [True, False])
def test_muon_param_groups(tie):
    model = TinyLM(tie)
    groups = group_ids(muon_param_groups(model))
    assert groups[True] == {id(model.proj.weight)}
    expected_adamw = {model.embed.weight, model.proj.bias, model.norm.weight, model.norm.bias, model.lm_head.weight}
    assert groups[False] == {id(p) for p in expected_adamw}


def test_muon_param_groups_override():
    model = TinyLM(tie=False)
    model.proj.weight.use_muon = False
    model.lm_head.weight.use_muon = True
    groups = group_ids(muon_param_groups(model))
    assert groups[True] == {id(model.lm_head.weight)}
    assert id(model.proj.weight) in groups[False]


def test_muon_rejects_non_matrix_params():
    with pytest.raises(ValueError, match="only supports 2D"):
        Muon([nn.Parameter(torch.randn(8))])
    # A group without use_muon defaults to Muon, so a bias passed by mistake is caught.
    with pytest.raises(ValueError, match="only supports 2D"):
        Muon(nn.Linear(4, 4).parameters())


def test_muon_rejects_sharded_gradients():
    # LowLevelZeRO hands the inner optimizer flat 1D slices of each parameter.
    opt = Muon([nn.Parameter(torch.randn(4, 4))])
    shard = nn.Parameter(torch.randn(8))
    shard.grad = torch.randn(8)
    opt.param_groups[0]["params"] = [shard]
    with pytest.raises(RuntimeError, match="full 2D gradient"):
        opt.step()


def test_muon_state_dict_round_trip():
    model = TinyLM(tie=False)
    opt = Muon(muon_param_groups(model), lr=LR)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state_dict()
    # Hyperparameters stay plain data so JSON-based checkpoint formats can store them.
    json.dumps([{k: v for k, v in g.items() if k != "params"} for g in state["param_groups"]])

    restored = Muon(muon_param_groups(model), lr=LR)
    restored.load_state_dict(copy.deepcopy(state))
    for p in model.parameters():
        for key, value in opt.state[p].items():
            torch.testing.assert_close(restored.state[p][key], value, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# DDP: two ranks on half batches must match one device on the full batch
# ---------------------------------------------------------------------------

DIM, GLOBAL_BATCH = 32, 8


def make_mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(DIM, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, DIM))


def train(model, batches, booster=None, rank=0, world_size=1):
    device = torch.device("cuda", torch.cuda.current_device())
    model = model.to(device)
    start = {n: p.detach().double().clone() for n, p in model.named_parameters()}
    opt = Muon(muon_param_groups(model), lr=LR, weight_decay=0.1)
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


def run_ddp(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    batches = torch.randn(STEPS, GLOBAL_BATCH, DIM, generator=torch.Generator().manual_seed(2))
    booster = Booster(plugin=TorchDDPPlugin())

    test = train(make_mlp(), batches, booster, rank, world_size)
    ref = train(make_mlp(), batches)
    # Same code at the same precision, so there is no baseline error to measure: the tool's relative bound applies.
    judge_relative_bound(test, ref, torch.bfloat16).assert_pass()
    # Each rank stepping on its own half batch (no gradient all-reduce) must be caught.
    unsynced = train(make_mlp(), batches, rank=rank, world_size=world_size)
    assert not judge_relative_bound(unsynced, ref, torch.bfloat16).passed


@pytest.mark.dist
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@rerun_if_address_is_in_use()
def test_muon_ddp():
    spawn(run_ddp, nprocs=2)


if __name__ == "__main__":
    test_muon_ddp()
