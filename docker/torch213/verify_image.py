"""Validate the dependency image without importing a checkout or accessing GPUs."""

import importlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def main():
    assert sys.version_info[:2] == (3, 10), sys.version
    root = Path(__file__).resolve().parent
    versions = {}
    for lock_name in ("requirements.lock", "requirements-torch.lock"):
        for line in (root / lock_name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            requirement = Requirement(line.split(" --hash=", 1)[0])
            actual = metadata.version(requirement.name)
            assert Version(actual) in requirement.specifier, (requirement.name, actual, str(requirement.specifier))
            versions[requirement.name] = actual

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for executable in ("nvcc", "g++", "cmake", "ninja", "git"):
        assert shutil.which(executable), f"Missing build tool: {executable}"
    nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
    assert "release 13.0" in nvcc, nvcc

    # A base image must not silently supply a stale ColossalAI package or AOT .so.
    try:
        metadata.distribution("colossalai")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError("Install ColossalAI from the CI source SHA at runtime, not in this image")

    for module in ("transformers", "peft", "diffusers", "triton", "testmon.pytest_testmon"):
        importlib.import_module(module)

    import torch
    import torchaudio
    import torchvision

    assert torch.version.cuda == "13.0", torch.version.cuda
    # CPU extension smoke checks catch some ABI mismatches that pip check cannot.
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    assert torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
    waveform = torch.zeros(1, 160)
    assert torchaudio.functional.resample(waveform, 16000, 8000).shape == (1, 80)
    # pytest --help loads installed entry points, including pytest-testmon.
    subprocess.run([sys.executable, "-m", "pytest", "--help"], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({"python": sys.version, "torch_cuda": torch.version.cuda, "packages": versions}, indent=2))


if __name__ == "__main__":
    main()
