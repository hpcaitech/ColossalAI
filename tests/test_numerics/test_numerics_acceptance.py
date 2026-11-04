import json
from types import SimpleNamespace

import torch
from torch import nn

from colossalai.accelerator import get_accelerator
from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.acceptance import format_report, run_acceptance
from tests.kit.numerics.scenarios.resume import unsharded_optimizer_reloads


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


def test_acceptance_report():
    report = run_acceptance(
        ENTRY, configs=["ddp_fp32", "ddp_bf16", "zero2_fp32"], checks=["blank_control", "independent_impl", "resume"]
    )
    status = {(r["check"], r["config"]): r["status"] for r in report["results"]}
    assert status.pop(("blank_control", None)) == "passed"
    assert status.pop(("independent_impl", "ddp_fp32")) == "passed"
    # on CPU the optimizer is saved unsharded, which an older torch cannot load back: resume does not run there
    no_resume = get_accelerator().name == "cpu" and not unsharded_optimizer_reloads()
    assert status.pop(("resume", "ddp_fp32")) == ("not_run" if no_resume else "passed")
    # BF16 autocast is implemented by the CUDA/NPU accelerator; CPUAccelerator has no BF16 autocast.
    on_device = "passed" if get_accelerator().name != "cpu" else "not_run"
    assert status.pop(("independent_impl", "ddp_bf16")) == on_device
    assert status.pop(("resume", "ddp_bf16")) == on_device
    # fp32, not bf16: LowLevelZero's bf16 resume is not yet bit-exact, which this report is right to flag
    assert status == {("independent_impl", "zero2_fp32"): on_device, ("resume", "zero2_fp32"): on_device}

    statuses = [r["status"] for r in report["results"]]
    assert report["summary"] == {s: statuses.count(s) for s in report["summary"]} and report["repository"]["commit"]
    passed = next(r for r in report["results"] if r["check"] == "independent_impl" and r["config"] == "ddp_fp32")
    assert passed["l2_item"] == "① ⑤" and passed["failures"] == [] and passed["tensors"] > 0
    json.loads(json.dumps(report))
    assert "unsupported" in format_report(report)
