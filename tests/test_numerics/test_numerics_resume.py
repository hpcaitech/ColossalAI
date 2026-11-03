from functools import partial
from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn

from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.testing import rerun_if_address_is_in_use
from tests.kit.numerics import NumericsSpec, judge_equal
from tests.kit.numerics.scenarios import resume

# loading a sharded optimizer fails on CPU (load_states_into_optimizer calls the CPU accelerator's synchronize,
# which raises), so the CPU tests save it unsharded; ColossalAI loads that back only with a recent enough torch
CPU = dict(world_size=1, shard_optimizer=False)
unsharded = pytest.mark.skipif(
    not resume.unsharded_optimizer_reloads(),
    reason="ColossalAI cannot load back its unsharded optimizer checkpoints with this torch (needs torch >= 2.10)",
)


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


def _tiny_llama():
    config = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        vocab_size=128,
        attn_implementation="eager",
    )
    return transformers.LlamaModel(config)


def _tokens():
    return {"input_ids": torch.randint(0, 128, (2, 8))}


def _hidden(output):
    return {"hidden": output.last_hidden_state}


# module-level parts only: the entries are sent to spawned processes
MLP = (TinyMLP, _data, _identity, None, _attribute(NumericsSpec()))
LLAMA = (_tiny_llama, _tokens, _hidden, None, _attribute(NumericsSpec()))
RESUMED = {"loss/2", "loss/3"} | {f"param/fc{i}.{p}" for i in (1, 2) for p in ("weight", "bias")}
# the optimizer's tensors (here the parameters themselves, in fc1.weight, fc1.bias, fc2.weight, fc2.bias order) and
# Adam's state, the scheduler's state and the random number generators
OPTIMIZER = {f"optim/0.{i}/{key}" for i in range(4) for key in ("param", "exp_avg", "exp_avg_sq", "step")}


@unsharded
@rerun_if_address_is_in_use()
def test_resume_is_exact():
    report = resume.check(MLP, TorchDDPPlugin, **CPU)
    names = {e.name for e in report.entries}
    assert RESUMED | OPTIMIZER | {"hf/<from_pretrained>", "rng/cpu", "scheduler/last_epoch"} <= names
    assert all(n.startswith(("scheduler/", "rng/")) for n in names - RESUMED - OPTIMIZER - {"hf/<from_pretrained>"})
    report.assert_pass()


# what a missing part changes besides the final weights, read directly
LOST = {
    "model": "optim/0.0/param",
    "optimizer": "optim/0.0/exp_avg",
    "lr_scheduler": "scheduler/last_epoch",
}


@unsharded
@pytest.mark.parametrize("missing", ["model", "optimizer", "lr_scheduler"])
@rerun_if_address_is_in_use()
def test_missing_checkpoint_part_detected(missing):
    parts = [p for p in ("model", "optimizer", "lr_scheduler") if p != missing]
    report = resume.check(MLP, TorchDDPPlugin, parts=parts, **CPU)
    # the weights after the resumed steps always differ; the losses only if the forward already sees the loss
    failed = {e.name for e in report.failures}
    assert {f"param/fc{i}.weight" for i in (1, 2)} | {LOST[missing]} <= failed, report.format()


class _DrawsNoise(TinyMLP):
    def forward(self, x):
        # random numbers drawn in training (here without effect on the result): a booster checkpoint does not carry
        # the generator state, so a resumed run continues from another one
        return super().forward(x + 0.0 * torch.rand_like(x))


@unsharded
@rerun_if_address_is_in_use()
def test_random_numbers_drawn_in_training_detected():
    report = resume.check((_DrawsNoise, _data, _identity, None, _attribute(NumericsSpec())), TorchDDPPlugin, **CPU)
    # drawn on the device the model runs on
    generator = "rng/cpu" if get_accelerator().name == "cpu" else "rng/cuda"
    assert [e.name for e in report.failures] == [generator], report.format()


@unsharded
@rerun_if_address_is_in_use()
def test_huggingface_round_trip():
    report = resume.check(LLAMA, TorchDDPPlugin, **CPU)
    exported = [e for e in report.entries if e.name.startswith("hf/param/")]
    assert len(exported) == sum(1 for _ in _tiny_llama().parameters())
    report.assert_pass()


@pytest.mark.skipif(
    get_accelerator().name == "cpu" or get_accelerator().device_count() < 2,
    reason="LowLevelZero requires two CUDA or NPU devices",
)
@rerun_if_address_is_in_use()
def test_low_level_zero_fp32_resume_is_exact():
    # sharded optimizer, as in real training; in bf16 the resumed run is not exact (see the README)
    plugin_fn = partial(LowLevelZeroPlugin, stage=2, precision="fp32")
    resume.check(MLP, plugin_fn, world_size=2).assert_pass()


def test_judge_equal():
    a = {"x": torch.arange(4.0), "n": torch.arange(4)}
    assert judge_equal(a, {"x": torch.arange(4.0), "n": torch.arange(4)}).passed
    report = judge_equal({"x": torch.arange(4.0) + 1e-6, "n": torch.arange(4)}, a)
    assert [e.name for e in report.failures] == ["x"] and report.failures[0].test_error > 0
    # the same values in another dtype are not the same run
    report = judge_equal({"x": torch.arange(4.0, dtype=torch.float64), "n": torch.arange(4)}, a)
    assert [e.name for e in report.failures] == ["x"]
