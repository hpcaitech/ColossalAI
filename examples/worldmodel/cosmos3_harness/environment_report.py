r"""Record the Cosmos 3 runtime baseline as JSON.

Versions alone do not identify the code that runs: a checkout can carry a patched ``diffusers`` while reporting the
same ``__version__``. This report records the distribution version, the module path, a content hash for the files that
implement Cosmos attention, the checkpoint revision if given, and the capabilities the pinned Torch build exposes.

    python examples/worldmodel/cosmos3_harness/environment_report.py --output logs/cosmos3-env.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

__all__ = ["collect_environment", "module_digest", "main"]


def module_digest(module) -> Dict:
    """Return the module's file path and SHA-256 digest, or an explicit absence marker."""
    path = getattr(module, "__file__", None)
    if not path:
        return {"file": None, "sha256": None}
    file_path = Path(path)
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return {"file": str(file_path), "sha256": digest}


def collect_environment(model: Optional[str] = None, revision: Optional[str] = None) -> Dict:
    """Collect the package, path, capability and checkpoint facts both devices must match."""
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "packages": {},
        "cosmos3": {},
    }
    for name in ("transformers", "diffusers", "colossalai", "torch_npu"):
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            report["packages"][name] = {"installed": False, "error": str(error)}
            continue
        report["packages"][name] = {
            "installed": True,
            "version": getattr(module, "__version__", None),
            **module_digest(module),
        }

    try:
        from torch.nn.attention.varlen import varlen_attn

        report["native_varlen"] = {
            "available": True,
            "signature": str(__import__("inspect").signature(varlen_attn)),
        }
    except Exception as error:
        report["native_varlen"] = {"available": False, "error": f"{type(error).__name__}: {error}"}

    try:
        diffusers = importlib.import_module("diffusers")
        transformer_cls = getattr(diffusers, "Cosmos3OmniTransformer", None)
        report["cosmos3"]["transformer_class"] = transformer_cls is not None
        if transformer_cls is not None:
            report["cosmos3"]["transformer_module"] = transformer_cls.__module__
            report["cosmos3"]["transformer_qualname"] = transformer_cls.__qualname__
            try:
                source = __import__("inspect").getsourcefile(transformer_cls)
                report["cosmos3"]["source_file"] = source
                report["cosmos3"]["source_sha256"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            except (TypeError, OSError) as error:
                report["cosmos3"]["source_error"] = str(error)
    except ImportError as error:
        report["cosmos3"]["available"] = False
        report["cosmos3"]["error"] = str(error)

    report["checkpoint"] = {"model": model, "revision": revision}
    return report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", default="cosmos3_environment.json")
    args = parser.parse_args(argv)

    report = collect_environment(args.model, args.revision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cosmos3_class": report["cosmos3"].get("transformer_class")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
