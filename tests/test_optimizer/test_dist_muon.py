import os
import tempfile
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchFSDPPlugin
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import Muon, OrScale
from colossalai.nn.optimizer.muon import zeropower_via_newtonschulz
from colossalai.testing import free_port, rerun_if_address_is_in_use, spawn
from tests.kit.numerics import NumericsSpec, Trace, judge_equal, judge_relative_bound
from tests.kit.numerics.capture.training import prepare, run_steps
from tests.kit.numerics.scenarios import resume

STEPS, LR = 3, 0.02
# odd sizes, so that ZeRO pads every matrix and the padding sits inside the last rank's slice
DIM_IN, HIDDEN, DIM_OUT = 31, 63, 17


def requires_gpus(n):
    return pytest.mark.skipif(torch.cuda.device_count() < n, reason=f"needs {n} GPUs")


def muon_optimizer(params, cls=Muon, **kwargs):
    params = list(params)
    groups = [
        {"params": [p for p in params if p.ndim == 2], "use_muon": True},
        {"params": [p for p in params if p.ndim != 2], "use_muon": False},
    ]
    return cls(groups, lr=LR, weight_decay=0.1, **kwargs)


ORSCALE = {"orscale": dict(variant="orscale"), "orscale_lm": dict(variant="orscale_lm", r_min=0.1, r_max=5.0)}


class ShardLocalMuon(Muon):
    """A typical sharding bug: each rank orthogonalizes its own slice as a 1 x n matrix."""

    def _orthogonalize_shards(self, shards, inputs, group):
        return [
            zeropower_via_newtonschulz(x.view(1, -1), group["ns_coefficients"], group["ns_steps"], group["eps"])
            .flatten()
            .to(p.device)
            for p, x in zip(shards, inputs)
        ]


class ShardLocalTrustRatio(OrScale):
    """A typical sharding bug: each rank takes the trust ratio from the norms of its own slice."""

    def _apply_shard_updates(self, shards, updates, group):
        for p, update in zip(shards, updates):
            shape = self._working_param(p).shape
            scale = self._shape_scale(shape, group)
            step = group["weight_decay"] * p.float() + scale * update.float()
            ratio = self._trust_ratio(p, p.norm().float(), step.norm(), shape, group)
            self._scaled_step(p, update, ratio, scale, group)


# ---------------------------------------------------------------------------
# Muon and OrScale on hand-made ZeRO shards, over gloo: runs without GPUs
# ---------------------------------------------------------------------------

# name: (optimizer on the shards, optimizer on the whole matrices it must match, their arguments, bound).
# Muon's sharded step gathers the bf16 cast the whole-matrix step makes, so the results are the same bits (no bound).
# OrScale's trust ratio comes from norms summed slice by slice, which round differently from the norm of the whole
# matrix: its weights agree within the float32 relative bound. The gradients are given, so nothing amplifies that.
SHARD_CASES = {
    "muon": (Muon, Muon, {}, None),
    "orscale": (OrScale, OrScale, ORSCALE["orscale"], torch.float32),
    "orscale_lm": (OrScale, OrScale, ORSCALE["orscale_lm"], torch.float32),
}
BUGGY_SHARD_CASES = {
    "muon": (ShardLocalMuon, Muon, {}, torch.bfloat16),
    "orscale": (ShardLocalTrustRatio, OrScale, ORSCALE["orscale"], torch.float32),
    "orscale_lm": (ShardLocalTrustRatio, OrScale, ORSCALE["orscale_lm"], torch.float32),
}


def shard_of(t, world_size, rank):
    """``rank``'s slice of ``t``, padded and split the way LowLevelZeRO does."""
    flat = t.flatten()
    flat = F.pad(flat, (0, (-flat.numel()) % world_size))
    return flat.view(world_size, -1)[rank]


def run_shards(rank, world_size, port, test_cls, ref_cls, kwargs, bound, buggy):
    # colossalai.launch would pick NCCL on a GPU machine; these shards live on the CPU
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    gen = torch.Generator().manual_seed(0)
    shapes = {"tall": (HIDDEN, DIM_OUT), "wide": (DIM_IN, HIDDEN), "square": (48, 48), "bias": (HIDDEN,)}
    # weights about as large as their orthogonalized updates, so that OrScale's trust ratios are near 1, not clipped
    init = {n: 0.125 * torch.randn(s, generator=gen) for n, s in shapes.items()}
    grads = [{n: torch.randn(s, generator=gen) for n, s in shapes.items()} for _ in range(STEPS)]
    matrices = [n for n, s in shapes.items() if len(s) == 2]

    def groups(params):
        return [
            {"params": [params[n] for n in matrices], "use_muon": True},
            {"params": [params["bias"]], "use_muon": False},
        ]

    whole = {n: nn.Parameter(t.clone()) for n, t in init.items()}
    ref = ref_cls(groups(whole), lr=LR, weight_decay=0.1, **kwargs)
    # as with LowLevelZeRO: built on the whole parameters, then handed the shards and told what they belong to
    working = {n: nn.Parameter(t.clone()) for n, t in init.items()}
    shards = {n: nn.Parameter(shard_of(t, world_size, rank).clone()) for n, t in init.items()}
    test = test_cls(groups(working), lr=LR, weight_decay=0.1, **kwargs)
    for group, sharded in zip(test.param_groups, groups(shards)):
        group["params"] = sharded["params"]
    test.setup_distributed(None, None, {id(shards[n]): working[n] for n in shapes}, None, True)

    for step_grads in grads:
        for n in shapes:
            whole[n].grad = step_grads[n].clone()
            shards[n].grad = shard_of(step_grads[n], world_size, rank).clone()
        ref.step()
        test.step()

    def gathered(t, shape):
        parts = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(parts, t.detach().contiguous())
        return torch.cat(parts)[: torch.Size(shape).numel()].view(shape)

    final = {n: gathered(shards[n], shapes[n]) for n in shapes}
    # elementwise on the shards, so the same bits whatever the optimizer does with whole matrices
    exact = {f"momentum/{n}": gathered(test.state[shards[n]]["momentum_buffer"], shapes[n]) for n in matrices}
    exact["weight/bias"] = final["bias"]
    exact_expected = {f"momentum/{n}": ref.state[whole[n]]["momentum_buffer"] for n in matrices}
    exact_expected["weight/bias"] = whole["bias"].detach()
    judge_equal(exact, exact_expected).assert_pass()

    if bound == torch.float32:
        # The weights: the difference of two float32 weights does not resolve a step to 1e-6 of itself, while a wrong
        # trust ratio moves the weights by far more than 1e-6 of themselves.
        result = {f"matrix/{n}": final[n] for n in matrices}
        expected = {f"matrix/{n}": whole[n].detach() for n in matrices}
        if "c_denom" in ref.state[whole[matrices[0]]]:
            result.update({f"c_denom/{n}": torch.tensor(test.state[shards[n]]["c_denom"]) for n in matrices})
            expected.update({f"c_denom/{n}": torch.tensor(ref.state[whole[n]]["c_denom"]) for n in matrices})
        report = judge_relative_bound(result, expected, bound)
    else:
        # The updates: next to the weights a wrong update would hide inside the bf16 bound.
        result = {f"matrix/{n}": final[n] - init[n] for n in matrices}
        expected = {f"matrix/{n}": whole[n].detach() - init[n] for n in matrices}
        report = judge_equal(result, expected) if bound is None else judge_relative_bound(result, expected, bound)
    if buggy:
        assert {e.name for e in report.failures} >= {f"matrix/{n}" for n in matrices}, report.format()
    else:
        report.assert_pass()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
@pytest.mark.parametrize("case", SHARD_CASES)
def test_shards_match_whole_matrices(case, world_size):
    args = (world_size, free_port(), *SHARD_CASES[case], False)
    torch.multiprocessing.spawn(run_shards, args=args, nprocs=world_size)


@pytest.mark.parametrize("case", BUGGY_SHARD_CASES)
def test_shard_local_computation_detected(case):
    torch.multiprocessing.spawn(run_shards, args=(2, free_port(), *BUGGY_SHARD_CASES[case], True), nprocs=2)


def run_setup_checks(rank, world_size, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    weight = nn.Parameter(torch.randn(4, 6))
    optimizer = Muon([weight])
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        optimizer.setup_distributed(tp_group=dist.group.WORLD)
    # not one of world_size equal slices, as with expert parameters ZeRO shards over their own group
    shard = nn.Parameter(torch.randn(5))
    optimizer.param_groups[0]["params"] = [shard]
    with pytest.raises(ValueError, match="do not span the data-parallel group"):
        optimizer.setup_distributed(None, None, {id(shard): weight}, None, True)
    dist.destroy_process_group()


def test_setup_rejects_tensor_parallel_and_foreign_shards():
    torch.multiprocessing.spawn(run_setup_checks, args=(2, free_port()), nprocs=2)


# ---------------------------------------------------------------------------
# LowLevelZeRO on GPUs: N ranks against one rank of the same plugin
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, DIM_OUT)

    def forward(self, x):
        return {"y": self.fc2(self.norm(F.gelu(self.fc1(x))))}


def _data():
    return {"x": torch.randn(8, DIM_IN)}


def _identity(output):
    return output


# module-level parts only: the entry is sent to spawned processes
ENTRY = (MLP, _data, _identity, None, SimpleNamespace(numerics=NumericsSpec()))


def run_zero(rank, world_size, port, plugin_fn, optimizer_fn, steps, path):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    # every rank trains on the same batch: averaging N equal gradients is exact for N = 2, so only sharding differs
    trace = run_steps(prepare(ENTRY, steps), plugin_fn(), 0, steps, optimizer_fn=optimizer_fn)
    if rank == 0:
        trace.save(path)
    dist.destroy_process_group()


def zero_trace(plugin_fn, world_size, optimizer_fn=muon_optimizer, steps=STEPS):
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trace.safetensors")
        spawn(run_zero, world_size, plugin_fn=plugin_fn, optimizer_fn=optimizer_fn, steps=steps, path=path)
        return Trace.load(path)


def comparable(test, world_size, ref):
    """Losses, weights and the optimizer's state (float32 master weights, momentum, AdamW moments, OrScale-LM's
    calibrated constants) of both runs.

    The optimizer's tensors are recorded per rank in ZeRO's flat layout; the slices of ``test`` are joined and the
    padding cut to the length of ``ref``'s, which one rank holds unpadded."""
    keep = lambda k: k.startswith(("loss/", "param/"))  # noqa: E731
    t, r = {k: v for k, v in test.items() if keep(k)}, {k: v for k, v in ref.items() if keep(k)}
    for key, value in ref.items():
        if key.startswith("optim/"):
            parts = [test[f"rank{i}/{key}"].flatten() for i in range(world_size)]
            r[key] = value.flatten()
            t[key] = torch.cat(parts)[: r[key].numel()]
    return t, r


def updates(state, precision):
    """The master weights' change over the run instead of the weights, plus the optimizer's other state.

    For a relative bound: three steps at lr = 0.02 change these weights by a few percent, so a wrong update would stay
    within the bound if the weights themselves were compared."""
    params = [p for p in prepare(ENTRY, STEPS).model.parameters() if p.requires_grad]
    groups = [[p for p in params if p.ndim == 2], [p for p in params if p.ndim != 2]]  # as muon_optimizer builds them
    # the plugin casts the model to its precision before ZeRO copies the float32 master weights
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    start = {
        f"optim/{g}.{i}/param": p.detach().to(dtype).float().flatten()
        for g, ps in enumerate(groups)
        for i, p in enumerate(ps)
    }
    out = {}
    for key, value in state.items():
        if key in start:
            out[f"update/{key}"] = value - start[key]
        elif key.startswith("optim/"):
            out[key] = value
    return out


def optimizer_state(state):
    """What the optimizer updates and keeps: the float32 master weights, the momentum, OrScale-LM's constants."""
    return {key: value for key, value in state.items() if key.startswith("optim/")}


ZERO_CONFIGS = [(1, "fp32", False), (2, "fp32", False), (1, "bf16", False), (2, "bf16", False), (2, "bf16", True)]


@pytest.mark.dist
@requires_gpus(2)
@pytest.mark.parametrize("stage, precision, cpu_offload", ZERO_CONFIGS)
@rerun_if_address_is_in_use()
def test_zero_two_ranks_equal_one(stage, precision, cpu_offload):
    plugin_fn = partial(LowLevelZeroPlugin, stage=stage, precision=precision, cpu_offload=cpu_offload)
    test, ref = comparable(zero_trace(plugin_fn, 2), 2, zero_trace(plugin_fn, 1))
    judge_equal(test, ref).assert_pass()


ZERO_OPTIMIZERS = {
    "muon": muon_optimizer,
    **{n: partial(muon_optimizer, cls=OrScale, **kw) for n, kw in ORSCALE.items()},
}


@pytest.mark.dist
@requires_gpus(4)
@pytest.mark.parametrize("optimizer", ZERO_OPTIMIZERS)
@rerun_if_address_is_in_use()
def test_zero_four_ranks_within_bound(optimizer):
    # averaging 4 equal gradients rounds, so the runs differ in the last bits: Megatron-LM's relative bound
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="bf16")
    optimizer_fn = ZERO_OPTIMIZERS[optimizer]
    test, ref = comparable(zero_trace(plugin_fn, 4, optimizer_fn), 4, zero_trace(plugin_fn, 1, optimizer_fn))
    judge_relative_bound(updates(test, "bf16"), updates(ref, "bf16"), torch.bfloat16).assert_pass()


@pytest.mark.dist
@requires_gpus(2)
@pytest.mark.parametrize("variant", ORSCALE)
@pytest.mark.parametrize("stage, precision, cpu_offload", [(1, "fp32", False), (2, "bf16", False), (2, "bf16", True)])
@rerun_if_address_is_in_use()
def test_zero_orscale_two_ranks_match_one(variant, stage, precision, cpu_offload):
    # One step: the gradients, the momentum and the orthogonalized updates of the two runs are the same bits, and the
    # trust ratios differ by float32 rounding, so the float32 bound holds for the master weights (see run_shards).
    # Over more steps the bf16 orthogonalization turns that rounding into bf16 rounding, which only the bf16 bound
    # covers (test_zero_four_ranks_within_bound).
    plugin_fn = partial(LowLevelZeroPlugin, stage=stage, precision=precision, cpu_offload=cpu_offload)
    optimizer_fn = ZERO_OPTIMIZERS[variant]
    two, one = zero_trace(plugin_fn, 2, optimizer_fn, steps=1), zero_trace(plugin_fn, 1, optimizer_fn, steps=1)
    test, ref = comparable(two, 2, one)
    judge_relative_bound(optimizer_state(test), optimizer_state(ref), torch.float32).assert_pass()


@pytest.mark.dist
@requires_gpus(2)
@rerun_if_address_is_in_use()
def test_zero_shard_local_orthogonalization_detected():
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="bf16")
    shard_local = partial(muon_optimizer, cls=ShardLocalMuon)
    test, ref = comparable(zero_trace(plugin_fn, 2, shard_local), 2, zero_trace(plugin_fn, 1))
    report = judge_relative_bound(updates(test, "bf16"), updates(ref, "bf16"), torch.bfloat16)
    assert {e.name for e in report.failures} >= {"update/optim/0.0/param", "update/optim/0.1/param"}, report.format()


@pytest.mark.dist
@requires_gpus(2)
@rerun_if_address_is_in_use()
def test_zero_shard_local_trust_ratio_detected():
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="bf16")
    shard_local = partial(muon_optimizer, cls=ShardLocalTrustRatio, **ORSCALE["orscale"])
    two = zero_trace(plugin_fn, 2, shard_local, steps=1)
    test, ref = comparable(two, 2, zero_trace(plugin_fn, 1, ZERO_OPTIMIZERS["orscale"], steps=1))
    report = judge_relative_bound(optimizer_state(test), optimizer_state(ref), torch.float32)
    # the master weights of the matrices and nothing else; after one warm-up step the smaller matrix is only a few
    # times outside the bound, so it is not required to fail
    failed = {e.name for e in report.failures}
    assert "optim/0.0/param" in failed and failed <= {"optim/0.0/param", "optim/0.1/param"}, report.format()


@pytest.mark.dist
@requires_gpus(2)
@pytest.mark.parametrize("optimizer, shard", [("muon", True), ("orscale_lm", True), ("orscale_lm", False)])
@rerun_if_address_is_in_use()
def test_zero_resume_is_exact(optimizer, shard):
    # fp32: in bf16 no optimizer resumes exactly, as LowLevelZero rebuilds the float32 master weights from the bf16
    # ones on load (tests/kit/numerics/README.md, "Resuming from a checkpoint"). OrScale-LM keeps a float per matrix
    # in its state, which both checkpoint layouts have to carry.
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="fp32")
    optimizer_fn = ZERO_OPTIMIZERS[optimizer]
    resume.check(ENTRY, plugin_fn, world_size=2, optimizer_fn=optimizer_fn, shard_optimizer=shard).assert_pass()


def run_plugins(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    for cls in (Muon, OrScale):
        for plugin in (GeminiPlugin(), TorchFSDPPlugin()):
            with pytest.raises(NotImplementedError, match=f"does not support {cls.__name__}"):
                model = MLP().cuda()
                Booster(plugin=plugin).boost(model, muon_optimizer(model.parameters(), cls))
        model = MLP().cuda()
        inner = muon_optimizer(model.parameters(), cls)
        _, optimizer, *_ = Booster(plugin=LowLevelZeroPlugin(stage=1)).boost(model, inner)
        # set up in place, not replaced: an LR scheduler built on `inner` before boost keeps driving it
        assert optimizer.optim is inner and inner.shard_to_working_param
    dist.destroy_process_group()


@pytest.mark.dist
@requires_gpus(1)
@rerun_if_address_is_in_use()
def test_plugins_with_muon_and_orscale():
    spawn(run_plugins, 1)
