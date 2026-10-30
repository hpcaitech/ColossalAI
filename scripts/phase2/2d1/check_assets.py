"""Read-only pinned-cache audit; no full model allocation or GPU use."""
import argparse
import csv
import hashlib
import json
import math
import os
import struct
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from check_remote_tiny import MODELS

ALL_MODELS = {"llada": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"), **MODELS}
SIZES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

def require(value, message):
    if not value:
        raise AssertionError(message)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def inspect_shard(path):
    with path.open("rb") as f:
        raw = f.read(8)
        require(len(raw) == 8, f"Truncated header: {path}")
        length = struct.unpack("<Q", raw)[0]
        require(0 < length < 100_000_000, f"Invalid header length: {path}")
        header = json.loads(f.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    require(tensors, f"No tensors: {path}")
    end = 0
    elements = 0
    dtypes = {}
    for name, spec in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"]):
        start, stop = spec["data_offsets"]
        count = math.prod(spec["shape"])
        require(spec["dtype"] in SIZES, f"Unhandled dtype: {spec['dtype']}")
        require(start == end and stop - start == count * SIZES[spec["dtype"]],
                f"Invalid tensor extent: {path.name}:{name}")
        end = stop
        elements += count
        dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + count
    require(path.stat().st_size == 8 + length + end, f"Truncated/trailing shard data: {path}")
    return tensors, {"name": path.name, "file_bytes": path.stat().st_size, "tensor_bytes": end,
                     "tensor_count": len(tensors), "stored_elements": elements, "dtype_elements": dtypes}

def probe(key, repo, revision, rows, full_hash):
    result = {"key": key, "repo_id": repo, "revision": revision, "passed": False, "stage": "snapshot"}
    try:
        path = Path(snapshot_download(repo, revision=revision, local_files_only=True))
        result["snapshot"] = str(path)
        require(path.name == revision, "Snapshot revision mismatch")
        raw = json.loads((path / "config.json").read_text())
        result["config"] = raw
        result["stage"] = "manifest"
        selected = [r for r in rows if r["repo"] == repo and r["revision"] == revision]
        require(selected, f"No fixed-revision manifest for {repo}")
        evidence = []
        for row in selected:
            target = path / row["path"]
            require(target.is_file(), f"Missing file/broken link: {target}")
            require(target.stat().st_size == int(row["bytes"]), f"Size mismatch: {target}")
            require(row["hub_check"] == "ok", f"Historical Hub hash check not OK: {target}")
            checked = full_hash or target.suffix != ".safetensors"
            if checked:
                require(sha256(target) == row["sha256"], f"SHA256 mismatch: {target}")
            evidence.append({"path": row["path"], "bytes": int(row["bytes"]),
                             "manifest_sha256": row["sha256"], "sha256_recomputed": checked,
                             "historical_hub_check": row["hub_check"]})
        result["manifest_files"] = evidence
        result["stage"] = "weight_headers"
        indices = list(path.glob("*.safetensors.index.json"))
        require(len(indices) <= 1, "Multiple weight indexes require explicit selection")
        index = json.loads(indices[0].read_text()) if indices else None
        names = sorted(set(index["weight_map"].values())) if index else sorted(p.name for p in path.glob("*.safetensors"))
        require(names, "No safetensors weights")
        all_names = set()
        shard_reports = []
        for name in names:
            tensors, report = inspect_shard(path / name)
            require(not all_names.intersection(tensors), "Duplicated tensors across shards")
            all_names.update(tensors)
            if index:
                expected = {k for k, v in index["weight_map"].items() if v == name}
                require(set(tensors) == expected, f"Index/header key mismatch: {name}")
            shard_reports.append(report)
        if index:
            require(set(index["weight_map"]) == all_names, "Missing indexed tensors")
            require(sum(r["tensor_bytes"] for r in shard_reports) == index["metadata"]["total_size"],
                    "Index total_size mismatch")
        result["shards"] = shard_reports
        result["stored_elements"] = sum(r["stored_elements"] for r in shard_reports)
        result["tensor_bytes"] = sum(r["tensor_bytes"] for r in shard_reports)
        result["stage"] = "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
        ids = tokenizer.encode("Hello, 世界。2D1 probe.", add_special_tokens=False)
        require(ids and min(ids) >= 0 and max(ids) < raw["vocab_size"], "Invalid encoded token IDs")
        require(max(tokenizer.get_vocab().values()) < raw["vocab_size"], "Tokenizer exceeds embedding vocabulary")
        decoded = tokenizer.decode(ids)
        require(bool(decoded.strip()), "Empty decoded text")
        result["tokenizer"] = {"class": type(tokenizer).__name__, "length": len(tokenizer),
                               "sample_ids": ids, "decoded": decoded,
                               "special_tokens_map": tokenizer.special_tokens_map}
        result.update(passed=True, stage="complete")
        print(f"ASSETS_OK {key}: {len(selected)} files, {len(names)} shards, "
              f"{result['stored_elements']:,} stored elements; TOKENIZER_OK", flush=True)
    except Exception as e:
        result.update(error_type=type(e).__name__, error=str(e), traceback=traceback.format_exc())
        traceback.print_exc()
        print("ASSETS_FAIL", key, result["stage"], flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sha256", action="store_true", help="Also re-read all weight bytes; substantial shared-cache I/O")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = Path("/mnt/beegfs/ColossalAI/shared-cache/manifest/MANIFEST.tsv")
    with manifest.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    env = {"torch": torch.__version__, "transformers": transformers.__version__,
           "manifest": str(manifest), "manifest_sha256": sha256(manifest),
           "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    for name in ("memory.max", "cpu.max"):
        p = Path("/sys/fs/cgroup") / name
        env[name] = p.read_text().strip() if p.exists() else "unavailable"
    try:
        command = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        env["gpu_inventory"] = {"exit_code": command.returncode, "stdout": command.stdout, "stderr": command.stderr}
    except Exception as e:
        env["gpu_inventory"] = {"error": str(e)}
    print("ENVIRONMENT", json.dumps(env, ensure_ascii=False), flush=True)
    results = [probe(key, repo, revision, rows, args.full_sha256) for key, (repo, revision) in ALL_MODELS.items()]
    report = args.report or Path(os.environ.get("D2D1_LOG_DIR", "/results/2d1")) / (
        "assets-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as f:
        json.dump({"environment": env, "full_sha256": args.full_sha256, "results": results,
                   "scope": "Manifest/file sizes + current nonweight SHA256 + shard headers/index + offline tokenizer. "
                            "Weights use historical Hub hashes unless --full-sha256. No real-weight model execution."},
                  f, indent=2, ensure_ascii=False)
    print("JSON report:", report)
    return int(any(not r["passed"] for r in results))

if __name__ == "__main__":
    raise SystemExit(main())
