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
from colossalai.nn.optimizer import Muon
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


def muon_optimizer(params, cls=Muon):
    params = list(params)
    groups = [
        {"params": [p for p in params if p.ndim == 2], "use_muon": True},
        {"params": [p for p in params if p.ndim != 2], "use_muon": False},
    ]
    return cls(groups, lr=LR, weight_decay=0.1)


class ShardLocalMuon(Muon):
    """A typical sharding bug: each rank orthogonalizes its own slice as a 1 x n matrix."""

    def _orthogonalize_shards(self, shards, inputs, group):
        return [
            zeropower_via_newtonschulz(x.view(1, -1), group["ns_coefficients"], group["ns_steps"], group["eps"])
            .flatten()
            .to(p.device)
            for p, x in zip(shards, inputs)
        ]


# ---------------------------------------------------------------------------
# Muon on hand-made ZeRO shards, over gloo: runs without GPUs
# ---------------------------------------------------------------------------


def shard_of(t, world_size, rank):
    """``rank``'s slice of ``t``, padded and split the way LowLevelZeRO does."""
    flat = t.flatten()
    flat = F.pad(flat, (0, (-flat.numel()) % world_size))
    return flat.view(world_size, -1)[rank]


def run_shards(rank, world_size, port, optimizer_cls):
    # colossalai.launch would pick NCCL on a GPU machine; these shards live on the CPU
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    gen = torch.Generator().manual_seed(0)
    shapes = {"tall": (HIDDEN, DIM_OUT), "wide": (DIM_IN, HIDDEN), "square": (48, 48), "bias": (HIDDEN,)}
    init = {n: torch.randn(s, generator=gen) for n, s in shapes.items()}
    grads = [{n: torch.randn(s, generator=gen) for n, s in shapes.items()} for _ in range(STEPS)]
    matrices = [n for n, s in shapes.items() if len(s) == 2]

    def groups(params):
        return [
            {"params": [params[n] for n in matrices], "use_muon": True},
            {"params": [params["bias"]], "use_muon": False},
        ]

    whole = {n: nn.Parameter(t.clone()) for n, t in init.items()}
    ref = Muon(groups(whole), lr=LR, weight_decay=0.1)
    # as with LowLevelZeRO: built on the whole parameters, then handed the shards and told what they belong to
    working = {n: nn.Parameter(t.clone()) for n, t in init.items()}
    shards = {n: nn.Parameter(shard_of(t, world_size, rank).clone()) for n, t in init.items()}
    test = optimizer_cls(groups(working), lr=LR, weight_decay=0.1)
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

    # updates, not weights: next to unit-variance weights a wrong update would hide inside any relative bound
    result = {f"update/{n}": gathered(shards[n], shapes[n]) - init[n] for n in shapes}
    result.update({f"momentum/{n}": gathered(test.state[shards[n]]["momentum_buffer"], shapes[n]) for n in matrices})
    expected = {f"update/{n}": whole[n].detach() - init[n] for n in shapes}
    expected.update({f"momentum/{n}": ref.state[whole[n]]["momentum_buffer"] for n in matrices})
    if optimizer_cls is Muon:
        # the gathered input is the bf16 cast the whole-matrix step makes, so the results are the same bits
        judge_equal(result, expected).assert_pass()
    else:
        report = judge_relative_bound(result, expected, torch.bfloat16)
        assert {e.name for e in report.failures} >= {f"update/{n}" for n in matrices}, report.format()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_shards_match_whole_matrices(world_size):
    torch.multiprocessing.spawn(run_shards, args=(world_size, free_port(), Muon), nprocs=world_size)


def test_shard_local_orthogonalization_detected():
    torch.multiprocessing.spawn(run_shards, args=(2, free_port(), ShardLocalMuon), nprocs=2)


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


def run_zero(rank, world_size, port, plugin_fn, optimizer_fn, path):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    # every rank trains on the same batch: averaging N equal gradients is exact for N = 2, so only sharding differs
    trace = run_steps(prepare(ENTRY, STEPS), plugin_fn(), 0, STEPS, optimizer_fn=optimizer_fn)
    if rank == 0:
        trace.save(path)
    dist.destroy_process_group()


def zero_trace(plugin_fn, world_size, optimizer_fn=muon_optimizer):
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "trace.safetensors")
        spawn(run_zero, world_size, plugin_fn=plugin_fn, optimizer_fn=optimizer_fn, path=path)
        return Trace.load(path)


def comparable(test, world_size, ref):
    """Losses, weights and the optimizer's state (float32 master weights, momentum, AdamW moments) of both runs.

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


ZERO_CONFIGS = [(1, "fp32", False), (2, "fp32", False), (1, "bf16", False), (2, "bf16", False), (2, "bf16", True)]


@pytest.mark.dist
@requires_gpus(2)
@pytest.mark.parametrize("stage, precision, cpu_offload", ZERO_CONFIGS)
@rerun_if_address_is_in_use()
def test_zero_two_ranks_equal_one(stage, precision, cpu_offload):
    plugin_fn = partial(LowLevelZeroPlugin, stage=stage, precision=precision, cpu_offload=cpu_offload)
    test, ref = comparable(zero_trace(plugin_fn, 2), 2, zero_trace(plugin_fn, 1))
    judge_equal(test, ref).assert_pass()


@pytest.mark.dist
@requires_gpus(4)
@rerun_if_address_is_in_use()
def test_zero_four_ranks_within_bound():
    # averaging 4 equal gradients rounds, so the runs differ in the last bits: Megatron-LM's relative bound
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="bf16")
    test, ref = comparable(zero_trace(plugin_fn, 4), 4, zero_trace(plugin_fn, 1))
    judge_relative_bound(updates(test, "bf16"), updates(ref, "bf16"), torch.bfloat16).assert_pass()


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
def test_zero_resume_is_exact():
    # fp32: in bf16 no optimizer resumes exactly, as LowLevelZero rebuilds the float32 master weights from the bf16
    # ones on load (tests/kit/numerics/README.md, "Resuming from a checkpoint")
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="fp32")
    resume.check(ENTRY, plugin_fn, world_size=2, optimizer_fn=muon_optimizer).assert_pass()


def run_plugins(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    for plugin in (GeminiPlugin(), TorchFSDPPlugin()):
        with pytest.raises(NotImplementedError, match="does not support Muon"):
            model = MLP().cuda()
            Booster(plugin=plugin).boost(model, muon_optimizer(model.parameters()))
    model = MLP().cuda()
    muon = muon_optimizer(model.parameters())
    _, optimizer, *_ = Booster(plugin=LowLevelZeroPlugin(stage=1)).boost(model, muon)
    # set up in place, not replaced: an LR scheduler built on `muon` before boost keeps driving it
    assert optimizer.optim is muon and muon.shard_to_working_param
    dist.destroy_process_group()


@pytest.mark.dist
@requires_gpus(1)
@rerun_if_address_is_in_use()
def test_plugins_with_muon():
    spawn(run_plugins, 1)
