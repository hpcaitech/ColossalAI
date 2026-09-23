"""Run a small, explicit set of existing ColossalAI tests from this checkout."""

import argparse
import json
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = [
    "tests/test_booster/test_accelerator.py::test_accelerator",
    "tests/test_booster/test_plugin/test_dp_plugin_base.py::test_dp_plugin_dataloader",
]


def validate_report(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    expected = {node.split("::")[1] for node in TESTS}
    if len(cases) != len(TESTS) or {case.get("name") for case in cases} != expected:
        raise RuntimeError("Expected exactly the two selected ColossalAI tests")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise RuntimeError("ColossalAI tests failed or were skipped")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--gpus", nargs=2)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "gpu-h20-5":
        raise RuntimeError("This rollout targets gpu-h20-5")
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if not args.collect_only:
        import fcntl

        from pr_gpu import GPU_UUID

        if not args.gpus or len(set(args.gpus)) != 2 or not all(GPU_UUID.fullmatch(g) for g in args.gpus):
            raise RuntimeError("Two distinct GPU UUIDs are required")
        lock = open(f"/tmp/colossalai-e1-ricardoo-{os.getuid()}.lock", "a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for gpu in args.gpus:
            row = subprocess.check_output(
                ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                timeout=10,
            )
            memory, utilization = [value.strip() for value in row.split(",")]
            processes = subprocess.check_output(
                ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            if (
                not memory.isdigit()
                or not utilization.isdigit()
                or int(memory) > 256
                or int(utilization)
                or processes.strip()
            ):
                raise RuntimeError(f"Selected GPU is busy or its occupancy is unknown: {gpu}")

    os.environ.update(
        CUDA_VISIBLE_DEVICES="" if args.collect_only else ",".join(args.gpus),
        PYTHONPATH=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        NCCL_IB_DISABLE="1",
        NCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
        TORCH_EXTENSIONS_DIR=str(output / "torch_extensions"),
        TRITON_CACHE_DIR=str(output / "triton"),
        HF_HOME=str(output / "huggingface"),
    )
    os.chdir(root)
    sys.path.insert(0, str(root))
    import pytest
    import torch

    import colossalai

    if Path(colossalai.__file__).resolve() != root / "colossalai" / "__init__.py":
        raise RuntimeError("Imported ColossalAI is not from the tested source snapshot")
    if not torch.__version__.startswith("2.13."):
        raise RuntimeError("This initial suite requires the prevalidated Torch 2.13 environment")
    metadata = {
        "source": str(root),
        "colossalai_import": colossalai.__file__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tests": TESTS,
        "collect_only": args.collect_only,
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    if not args.collect_only and (not torch.cuda.is_available() or torch.cuda.device_count() != 2):
        raise RuntimeError("Exactly two CUDA devices must be visible")
    junit = output / "junit.xml"
    options = [*TESTS, "-v", "-ra", "--maxfail=1", "-p", "no:cacheprovider"]
    options += ["--collect-only"] if args.collect_only else [f"--junitxml={junit}"]
    code = int(pytest.main(options))
    if code:
        return code
    if not args.collect_only:
        count = validate_report(junit)
        print(json.dumps({"result": "E1_COLOSSALAI_PASS", "gpu_tested": True, "tests_passed": count}), flush=True)
    if lock is not None:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
