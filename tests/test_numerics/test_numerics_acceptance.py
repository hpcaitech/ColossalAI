import json

import torch
from torch import nn

from colossalai.accelerator import get_accelerator
from tests.kit.model_zoo.registry import ModelAttribute
from tests.kit.numerics import NumericsSpec
from tests.kit.numerics.acceptance import format_report, run_acceptance


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
ENTRY = (TinyMLP, _data, _identity, None, ModelAttribute(numerics=NumericsSpec()))


def test_acceptance_report():
    report = run_acceptance(
        ENTRY, configs=["ddp_fp32", "ddp_bf16", "zero2_fp32"], checks=["blank_control", "independent_impl", "resume"]
    )
    status = {(r["check"], r["config"]): r["status"] for r in report["results"]}
    assert status.pop(("blank_control", None)) == "passed"
    assert status.pop(("independent_impl", "ddp_fp32")) == "passed"
    assert status.pop(("resume", "ddp_fp32")) == "passed"
    # an unverified recipe is refused, not guessed; resume needs no recipe and hits the plugin's own error
    assert status.pop(("independent_impl", "ddp_bf16")) == "unsupported"
    assert status.pop(("resume", "ddp_bf16")) == "failed"
    # fp32, not bf16: LowLevelZero's bf16 resume is not yet bit-exact, which this report is right to flag
    on_device = "passed" if get_accelerator().name != "cpu" else "not_run"
    assert status == {("independent_impl", "zero2_fp32"): on_device, ("resume", "zero2_fp32"): on_device}

    assert report["summary"]["passed"] >= 3 and report["repository"]["commit"]
    passed = next(r for r in report["results"] if r["check"] == "independent_impl" and r["config"] == "ddp_fp32")
    assert passed["l2_item"] == "① ⑤" and passed["failures"] == [] and passed["tensors"] > 0
    json.loads(json.dumps(report))
    assert "unsupported" in format_report(report)
