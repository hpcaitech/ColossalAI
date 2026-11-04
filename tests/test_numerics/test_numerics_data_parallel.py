import itertools
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.numerics import NumericsSpec, TokenLoss, judge_relative_bound, relative_bound
from tests.kit.numerics.scenarios import data_parallel

GRADS = {f"grad/param/fc{i}.{p}" for i in (1, 2) for p in ("weight", "bias")}


def _attribute(spec):
    # all the tool reads from an entry's ModelAttribute; importing tests.kit.model_zoo would need timm, torchvision and
    # torchaudio, which the CPU CI job does not install
    return SimpleNamespace(numerics=spec)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        return {"y": self.fc2(torch.relu(self.fc1(x)))}


def _data():
    return {"x": torch.randn(16, 64)}


def _identity(output):
    return output


# module-level parts only: the entry is sent to spawned processes
ENTRY = (TinyMLP, _data, _identity, None, _attribute(NumericsSpec()))


class UnsyncedDDPPlugin(TorchDDPPlugin):
    """DDP with one parameter left out of gradient synchronization: a typical data-parallel bug."""

    def configure(self, model, *args, **kwargs):
        model._ddp_params_and_buffers_to_ignore = ["fc1.weight"]
        return super().configure(model, *args, **kwargs)


@rerun_if_address_is_in_use()
def test_ddp_fp32_equivalent():
    report = data_parallel.check(ENTRY, TorchDDPPlugin, world_size=2, micro_batches=4)
    assert {e.name for e in report.entries} == GRADS
    report.assert_pass()


@rerun_if_address_is_in_use()
def test_unsynchronized_parameter_detected():
    report = data_parallel.check(ENTRY, UnsyncedDDPPlugin, world_size=2)
    assert {e.name for e in report.failures} == {"grad/param/fc1.weight"}, report.format()
    # far beyond the bound, not a borderline call
    assert report.failures[0].ratio > 3, report.format()


@pytest.mark.skipif(
    get_accelerator().name == "cpu" or get_accelerator().device_count() < 2,
    reason="LowLevelZero requires two CUDA or NPU devices",
)
@rerun_if_address_is_in_use()
def test_low_level_zero_bf16_equivalent():
    data_parallel.check(ENTRY, partial(LowLevelZeroPlugin, stage=2, precision="bf16"), world_size=2).assert_pass()


def test_relative_bound_uses_megatron_constants():
    assert relative_bound(torch.bfloat16) == pytest.approx(4 * 1.03**33 * 2**-7)
    assert relative_bound(torch.bfloat16) == pytest.approx(0.0829, abs=1e-4)


def test_judge_relative_bound():
    g = torch.Generator().manual_seed(0)
    ref = {"a": torch.randn(256, generator=g), "b": torch.randn(256, generator=g)}
    close = {"a": ref["a"] * (1 + 1e-3), "b": ref["b"]}
    assert judge_relative_bound(close, ref, torch.bfloat16).passed
    # a wrong averaging factor
    report = judge_relative_bound({"a": ref["a"] * 2, "b": ref["b"]}, ref, torch.bfloat16)
    assert [e.name for e in report.failures] == ["a"] and report.failures[0].ratio > 10
    # tiny on both sides passes per tensor, but a comparison of nothing but zeros fails
    zeros = {"a": torch.zeros(4), "b": torch.full((4,), 1e-6)}
    report = judge_relative_bound(zeros, {"a": torch.zeros(4), "b": torch.zeros(4)}, torch.bfloat16)
    assert [e.name for e in report.failures] == ["<all>"]


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.head = nn.Linear(16, 32)

    def forward(self, input_ids, labels=None):
        # labels are for the loss; the model only returns logits
        return self.head(torch.tanh(self.embed(input_ids)))


def _padded_tokens():
    # a padded batch with a random number of valid tokens per sequence, so micro-batches differ in valid tokens
    input_ids = torch.randint(0, 32, (4, 12))
    labels = input_ids.clone()
    for row, length in enumerate(torch.randint(2, 13, (4,)).tolist()):
        labels[row, length:] = -100
    return {"input_ids": input_ids, "labels": labels}


def _summed_lm_loss(logits, batch):
    return F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), ignore_index=-100, reduction="sum")


def _valid_tokens(batch):
    return (batch["labels"] != -100).sum()


LM_ENTRY = (
    TinyLM,
    _padded_tokens,
    _identity,
    None,
    _attribute(NumericsSpec(token_loss=TokenLoss(loss=_summed_lm_loss, count=_valid_tokens))),
)


@rerun_if_address_is_in_use()
def test_token_weighted_equivalent():
    report = data_parallel.check(LM_ENTRY, TorchDDPPlugin, world_size=2, micro_batches=4, token_weighted=True)
    assert {e.name for e in report.entries} == {
        "grad/param/embed.weight",
        "grad/param/head.weight",
        "grad/param/head.bias",
    }
    report.assert_pass()


# batches drawn so far in this process; every process draws the same sequence, as prepare() draws them in order
_DRAWN = itertools.count()


def _one_empty_micro_batch():
    # the second of every four micro-batches has no valid token, e.g. sequences of padding only
    batch = _padded_tokens()
    if next(_DRAWN) % 4 == 1:
        batch["labels"][:] = -100
    return batch


def _no_tokens():
    batch = _padded_tokens()
    batch["labels"][:] = -100
    return batch


TOKEN_LOSS = _attribute(NumericsSpec(token_loss=TokenLoss(loss=_summed_lm_loss, count=_valid_tokens)))


@rerun_if_address_is_in_use()
def test_micro_batch_without_tokens():
    entry = (TinyLM, _one_empty_micro_batch, _identity, None, TOKEN_LOSS)
    kwargs = dict(world_size=2, micro_batches=4, token_weighted=True)
    for trace in data_parallel.traces(entry, TorchDDPPlugin, **kwargs):
        # its summed loss, not 0 / 0; and the rank holding it still took part in the gradient reduction
        assert trace["loss/0/1"].item() == 0.0
        assert all(torch.isfinite(v).all() for v in trace.values())
    data_parallel.check(entry, TorchDDPPlugin, **kwargs).assert_pass()


@rerun_if_address_is_in_use()
def test_step_without_tokens_rejected():
    entry = (TinyLM, _no_tokens, _identity, None, TOKEN_LOSS)
    with pytest.raises(Exception, match="no valid token"):
        data_parallel.check(entry, TorchDDPPlugin, world_size=2, micro_batches=4, token_weighted=True)


def _without_token_loss(rank, world_size, port):
    import colossalai
    from tests.kit.numerics.capture.training import prepare, run_data_parallel

    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port)
    with pytest.raises(ValueError, match="token_loss"):
        run_data_parallel(prepare(ENTRY, 2), TorchDDPPlugin(), 2, token_weighted=True)


@rerun_if_address_is_in_use()
def test_token_weighted_needs_token_loss():
    from colossalai.testing import spawn

    spawn(_without_token_loss, 1)


TWO_DEVICES = get_accelerator().name == "cpu" or get_accelerator().device_count() < 2


def _fsdp_bf16():
    from torch.distributed.fsdp import MixedPrecision

    from colossalai.booster.plugin import TorchFSDPPlugin

    bf16 = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return TorchFSDPPlugin(mixed_precision=bf16)


def _gemini_bf16(**kwargs):
    from colossalai.booster.plugin import GeminiPlugin

    return GeminiPlugin(precision="bf16", **kwargs)


# Gemini with one micro-batch per rank: with gradient accumulation on more than one device its gradients are wrong
# (the single-device side, accumulating, agrees with LowLevelZero; see the README), which is the plugin's to fix
@pytest.mark.skipif(TWO_DEVICES, reason="Gemini and FSDP require two CUDA devices")
@pytest.mark.parametrize(
    "plugin_fn, micro_batches",
    [(partial(_gemini_bf16, enable_gradient_accumulation=True), 2), (_fsdp_bf16, 4)],
    ids=["gemini", "fsdp"],
)
@rerun_if_address_is_in_use()
def test_sharded_plugins_equivalent(plugin_fn, micro_batches):
    report = data_parallel.check(ENTRY, plugin_fn, world_size=2, micro_batches=micro_batches)
    assert {e.name for e in report.entries} == GRADS
    report.assert_pass()


@pytest.mark.skipif(TWO_DEVICES, reason="Gemini requires two CUDA devices")
@rerun_if_address_is_in_use()
def test_gemini_accumulation_needs_its_flag():
    with pytest.raises(Exception, match="enable_gradient_accumulation"):
        data_parallel.check(ENTRY, _gemini_bf16, world_size=2, micro_batches=4)
