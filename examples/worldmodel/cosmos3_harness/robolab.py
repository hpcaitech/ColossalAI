"""Reproducible RoboLab DROID evaluation, without importing Isaac or an accelerator.

The simulator, joint-position controller, camera composition, action chunk cache and success
predicates belong to RoboLab. This adapter pins that implementation and invokes its Cosmos3
runner. GPU and NPU policy servers consume the same manifest; Isaac still runs on an RTX GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

INSTRUCTION_TYPES = ("vague", "default", "specific")
DIFFICULTIES = ("simple", "moderate", "complex")
SIMULATOR_STACKS = {
    "isaac50": {"isaac_sim": "5.0", "isaac_lab": "2.2.0"},
    "isaac51": {"isaac_sim": "5.1", "isaac_lab": "2.3.2.post1"},
    "isaac60": {"isaac_sim": "6.0.1", "isaac_lab": "3.0.0b2.post1"},
}
POLICY_CONFIG = {
    "mode": "wam",
    "action_space": "joint_pos",
    "action_dim": 8,
    "domain_id": 8,
    "chunk_size": 32,
    "fps": 15.0,
    "image_height": 540,
    "image_width": 640,
    "resolution": "480",
    "use_state": True,
    "guidance": 3.0,
    "num_steps": 4,
    "shift": 5.0,
    "sampler": "FlowUniPCMultistepScheduler",
    "seed": 0,
    # Matches the reference server's --deterministic-seed. Independent of request arrival order.
    "deterministic_seed": True,
    "noise": "numpy-randomstate-per-modality-float32",
    "format_prompt_as_json": True,
    "decode_video": False,
    "transformer_dtype": "bfloat16",
    "vae_dtype": "float32",
}


def fingerprint(payload: Mapping) -> str:
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(payload: Mapping, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # An interruption must not turn a previous receipt/summary into truncated JSON.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def checkpoint_fingerprint(checkpoint: Path, transformer_path: Path | None = None) -> str:
    """Hash loaded components by content, independently of host paths or HF snapshot symlinks."""
    entries = {}
    for component in ("transformer", "vae", "text_tokenizer"):
        root = transformer_path if component == "transformer" and transformer_path else checkpoint / component
        files = sorted(
            p for p in root.rglob("*") if p.is_file() and not any(x.startswith(".") for x in p.relative_to(root).parts)
        )
        if not files:
            raise ValueError(f"Missing checkpoint component: {root}")
        for file in files:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries[f"{component}/{file.relative_to(root).as_posix()}"] = digest.hexdigest()
    return fingerprint(entries)


def checkout_revision(root: str | Path) -> str:
    """Require a clean source checkout; generated output and ignored caches don't affect the pin."""
    root = Path(root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    # Results are normally under output/. They are artifacts, not imported Python sources.
    changes = [line for line in changes.splitlines() if not line[3:].startswith("output/")]
    if changes:
        raise ValueError(f"Evaluation source checkout is dirty: {root}\n" + "\n".join(changes))
    return revision


def task_catalog(root: str | Path) -> dict:
    """Read the official task metadata without importing tasks (which would start Isaac imports)."""
    root = Path(root)
    rows = json.loads((root / "robolab/tasks/_metadata/task_metadata.json").read_text(encoding="utf-8"))
    catalog = {}
    for row in rows:
        filename = row.get("filename", "")
        if not filename.startswith("benchmark/"):
            continue
        relative = Path(filename)
        if ".." in relative.parts:
            raise ValueError(f"Invalid task filename: {filename}")
        name = row["task_name"]
        if name in catalog:
            raise ValueError(f"Duplicate task in RoboLab metadata: {name}")
        if row.get("difficulty_label") not in DIFFICULTIES:
            raise ValueError(f"Missing official difficulty label for {name}")
        prompts = row.get("instruction_variants", {})
        if any(not isinstance(prompts.get(level), str) or not prompts[level].strip() for level in INSTRUCTION_TYPES):
            raise ValueError(f"Missing instruction variants for {name}")
        catalog[name] = {
            "filename": filename,
            "source_sha256": hashlib.sha256((root / "robolab/tasks" / relative).read_bytes()).hexdigest(),
            "difficulty": row["difficulty_label"],
            "instructions": {level: prompts[level] for level in INSTRUCTION_TYPES},
        }
    if not catalog:
        raise ValueError("No benchmark tasks found in RoboLab metadata")
    return catalog


def build_manifest(
    robolab_root: str | Path,
    cosmos_root: str | Path,
    *,
    model: str,
    model_revision: str,
    tasks: Sequence[str] | None = None,
    instruction_types: Sequence[str] = INSTRUCTION_TYPES,
    num_envs: int = 10,
    num_runs: int = 1,
    seed: int = 0,
    policy_seed: int = 0,
    simulator_stack: str = "isaac50",
) -> dict:
    if not model or not model_revision or model_revision in {"main", "latest", "pending-pin"}:
        raise ValueError("Provide a model and immutable model revision (or a local checkpoint digest)")
    if any(type(value) is not int or value < 1 for value in (num_envs, num_runs)) or any(
        type(value) is not int or not 0 <= value < 2**32 for value in (seed, policy_seed)
    ):
        raise ValueError("num_envs/num_runs must be positive integers and seeds must be uint32")
    levels = list(instruction_types)
    if not levels or len(set(levels)) != len(levels) or set(levels) - set(INSTRUCTION_TYPES):
        raise ValueError(f"instruction_types must be distinct members of {INSTRUCTION_TYPES}")
    catalog = task_catalog(robolab_root)
    names = sorted(catalog if tasks is None else tasks)
    if not names or len(set(names)) != len(names) or set(names) - set(catalog):
        raise ValueError("Tasks must be a nonempty, distinct subset of the official benchmark catalog")
    payload = {
        "schema_version": 1,
        "benchmark": "RoboLab-120" if len(names) == 120 and set(names) == set(catalog) else "RoboLab-subset",
        "model": model,
        "model_revision": model_revision,
        "sources": {
            "robolab": {"url": "https://github.com/NVlabs/RoboLab", "revision": checkout_revision(robolab_root)},
            "cosmos_framework": {
                "url": "https://github.com/NVIDIA/cosmos-framework",
                "revision": checkout_revision(cosmos_root),
            },
        },
        "tasks": {name: catalog[name] for name in names},
        "evaluation": {
            "instruction_types": levels,
            "num_envs": num_envs,
            "num_runs": num_runs,
            "seed": seed,
            "simulator": SIMULATOR_STACKS[simulator_stack],
            "renderer": "realtime",
            "rendering_type": "balanced",
            "enable_subtask": True,
        },
        "policy": {**POLICY_CONFIG, "seed": policy_seed},
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def load_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint(payload):
        raise ValueError("RoboLab manifest fingerprint mismatch; regenerate the manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported RoboLab manifest schema")
    expected_policy = {**POLICY_CONFIG, "seed": payload["policy"].get("seed")}
    if payload["policy"] != expected_policy:
        raise ValueError("Unsupported policy contract; this adapter serves the DROID WAM recipe")
    seed = payload["policy"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Policy seed must be uint32")
    return payload


def server_contract_matches(manifest: Mapping, metadata: Mapping, backend: str) -> bool:
    return (
        metadata.get("manifest_fingerprint") == manifest["fingerprint"]
        and metadata.get("backend") == backend
        and metadata.get("model_revision") == manifest["model_revision"]
        and metadata.get("policy") == manifest["policy"]
        and isinstance(metadata.get("checkpoint_fingerprint"), str)
        and re.fullmatch(r"[0-9a-f]{64}", metadata["checkpoint_fingerprint"]) is not None
    )


def verify_checkout(manifest: Mapping, root: str | Path, source: str) -> None:
    if checkout_revision(root) != manifest["sources"][source]["revision"]:
        raise ValueError(f"{source} checkout does not match the pinned manifest revision")
    if source == "robolab":
        catalog = task_catalog(root)
        if any(catalog.get(name) != task for name, task in manifest["tasks"].items()):
            raise ValueError("RoboLab task source or metadata differs from the manifest")


def evaluation_commands(
    manifest: Mapping,
    robolab_root: str | Path,
    *,
    python: str,
    remote_uri: str,
    run_name: str,
    video_mode: str = "all",
) -> list[tuple[str, list[str]]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name) or run_name in {".", ".."}:
        raise ValueError("run_name must be a single folder name containing letters, digits, '.', '_' or '-'")
    if not remote_uri.startswith(("ws://", "wss://")):
        raise ValueError("remote_uri must start with ws:// or wss://")
    evaluation = manifest["evaluation"]
    commands = []
    for level in evaluation["instruction_types"]:
        commands.append(
            (
                level,
                [
                    python,
                    str(Path(robolab_root).resolve() / "policies/cosmos3/run.py"),
                    "--remote-uri",
                    remote_uri,
                    "--task",
                    *manifest["tasks"],
                    "--instruction-type",
                    level,
                    "--num-envs",
                    str(evaluation["num_envs"]),
                    "--num-runs",
                    str(evaluation["num_runs"]),
                    "--seed",
                    str(evaluation["seed"]),
                    "--renderer",
                    evaluation["renderer"],
                    "--rendering-type",
                    evaluation["rendering_type"],
                    "--enable-subtask",
                    "--video-mode",
                    video_mode,
                    "--output-folder-name",
                    f"{run_name}/{level}",
                    "--headless",
                ],
            )
        )
    return commands


def run_evaluation(args, manifest: Mapping) -> None:
    verify_checkout(manifest, args.robolab_root, "robolab")
    commands = evaluation_commands(
        manifest,
        args.robolab_root,
        python=args.python,
        remote_uri=args.remote_uri,
        run_name=args.run_name,
        video_mode=args.video_mode,
    )
    if args.dry_run:
        print(json.dumps({"cwd": str(Path(args.robolab_root).resolve()), "commands": dict(commands)}, indent=2))
        return
    from examples.worldmodel.cosmos3_harness.robolab_transport import server_metadata

    versions = json.loads(
        subprocess.check_output(
            [
                args.python,
                "-c",
                "import importlib.metadata as m, json; "
                "print(json.dumps({n: m.version(n) for n in ['isaacsim', 'isaaclab']}))",
            ],
            text=True,
        )
    )
    for package, key in (("isaacsim", "isaac_sim"), ("isaaclab", "isaac_lab")):
        expected = manifest["evaluation"]["simulator"][key]
        if versions[package] != expected and not versions[package].startswith(expected + "."):
            raise ValueError(f"{package}={versions[package]} does not match pinned simulator stack {expected}")
    metadata = server_metadata(args.remote_uri)
    if not args.external_server or "manifest_fingerprint" in metadata:
        if not server_contract_matches(manifest, metadata, args.backend):
            raise ValueError("Policy endpoint manifest/backend does not match this evaluation")
    output = Path(args.robolab_root).resolve() / "output" / args.run_name
    receipt = {
        "manifest_fingerprint": manifest["fingerprint"],
        "backend": args.backend,
        "server_metadata": metadata,
        "external_server": args.external_server,
        "commands": dict(commands),
        "simulator": manifest["evaluation"]["simulator"],
        "simulator_versions": versions,
    }
    for level, command in commands:
        folder = output / level
        receipt_path = folder / "harness_run.json"
        if receipt_path.exists():
            if json.loads(receipt_path.read_text()) != receipt:
                raise ValueError(f"Cannot resume a run with a different configuration: {folder}")
        elif folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Refusing to reuse results without a harness receipt: {folder}")
        write_json(receipt, receipt_path)
        subprocess.run(command, cwd=Path(args.robolab_root).resolve(), check=True)
    write_json(summarize_results(manifest, output), output / "summary.json")


def _load_episodes(folder: Path) -> tuple[list, str | None]:
    path = folder / "episode_results.jsonl"
    if path.exists():
        # Do not silently skip truncated/malformed lines: they indicate an interrupted or corrupt run.
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        path = folder / "episode_results.json"
        if not path.exists():
            return [], None
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of episodes in {path}")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def _rates(rows: list[dict]) -> dict:
    tasks = sorted({row["task"] for row in rows})
    per_task = {}
    for task in tasks:
        outcomes = [row["success"] for row in rows if row["task"] == task]
        per_task[task] = {
            "successes": sum(outcomes),
            "episodes": len(outcomes),
            "success_rate": sum(outcomes) / len(outcomes),
        }
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "task_mean_success_rate": (
            sum(item["success_rate"] for item in per_task.values()) / len(tasks) if tasks else None
        ),
        "episode_success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None,
        "per_task": per_task,
    }


def summarize_results(manifest: Mapping, results_root: str | Path, *, allow_partial: bool = False) -> dict:
    """Score the official boolean success, never the subtask progress score.

    A complete run has exactly num_runs * num_envs episodes per task and instruction.
    Report task means and episode means separately; both agree for the balanced full benchmark.
    """
    evaluation = manifest["evaluation"]
    num_envs, num_runs = evaluation["num_envs"], evaluation["num_runs"]
    episodes, missing, digests, backends, receipts = [], [], {}, set(), {}
    provenance_verified = True
    for level in evaluation["instruction_types"]:
        folder = Path(results_root) / level
        receipt_path = folder / "harness_run.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["manifest_fingerprint"] != manifest["fingerprint"]:
                raise ValueError(f"Results were produced with a different manifest: {folder}")
            backends.add(receipt["backend"])
            receipts[level] = receipt
            metadata = receipt.get("server_metadata", {})
            provenance_verified &= server_contract_matches(manifest, metadata, receipt["backend"]) and bool(
                receipt.get("simulator_versions")
            )
        elif not allow_partial:
            raise ValueError(f"Missing harness_run.json in {folder}; provenance cannot be verified")
        else:
            provenance_verified = False
        rows, digests[level] = _load_episodes(folder)
        seen = set()
        for row in rows:
            task = row.get("task_name")
            if task not in manifest["tasks"] or row.get("env_name") != task:
                raise ValueError(f"Unexpected task/environment in {folder}: {task!r}")
            if type(row.get("success")) is not bool:
                raise ValueError(f"{task}: success must be an explicit boolean")
            run, env = row.get("run"), row.get("env_id")
            if type(run) is not int or type(env) is not int or not (0 <= run < num_runs and 0 <= env < num_envs):
                raise ValueError(f"{task}: out-of-range run/env_id")
            key = (task, run, env)
            if key in seen:
                raise ValueError(f"Duplicate episode in {folder}: {key}")
            seen.add(key)
            if type(row.get("episode")) is not int or row["episode"] != run * num_envs + env:
                raise ValueError(f"{task}: episode index disagrees with run/env_id")
            fields = row.get("instruction_fields_used", [row.get("instruction_type")])
            if fields != [level] or row.get("instruction_type", level) != level:
                raise ValueError(f"{task}: instruction type does not match {level}")
            if row.get("instruction") != manifest["tasks"][task]["instructions"][level]:
                raise ValueError(f"{task}: instruction text does not match the pinned catalog")
            dt = row.get("dt")
            if (
                row.get("policy") != "cosmos3"
                or type(dt) not in (int, float)
                or not math.isfinite(dt)
                or abs(dt - 1 / POLICY_CONFIG["fps"]) > 1e-8
            ):
                raise ValueError(f"{task}: expected Cosmos3 at 15 Hz")
            episodes.append(
                {"instruction_type": level, "task": task, "run": run, "env_id": env, "success": row["success"]}
            )
        missing.extend(
            {"instruction_type": level, "task": task, "run": run, "env_id": env}
            for task in manifest["tasks"]
            for run in range(num_runs)
            for env in range(num_envs)
            if (task, run, env) not in seen
        )
    if len(backends) > 1:
        raise ValueError("Results mix policy backends")
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete RoboLab evaluation: {len(missing)} episodes missing; use --allow-partial to inspect"
        )
    return {
        "schema_version": 1,
        "manifest_fingerprint": manifest["fingerprint"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "benchmark": manifest["benchmark"],
        "backend": next(iter(backends), None),
        "complete": not missing,
        "provenance_verified": bool(provenance_verified),
        "missing_episodes": missing,
        "result_sha256": digests,
        "run_receipts": receipts,
        "by_instruction": {
            level: {
                **_rates([row for row in episodes if row["instruction_type"] == level]),
                "by_difficulty": {
                    difficulty: _rates(
                        [
                            row
                            for row in episodes
                            if row["instruction_type"] == level
                            and manifest["tasks"][row["task"]]["difficulty"] == difficulty
                        ]
                    )
                    for difficulty in DIFFICULTIES
                },
            }
            for level in evaluation["instruction_types"]
        },
        "episodes": sorted(episodes, key=lambda row: (row["instruction_type"], row["task"], row["run"], row["env_id"])),
    }


def compare_results(left: Mapping, right: Mapping) -> dict:
    """Compare matched episode outcomes only after proving both runs used the same protocol."""
    if left["manifest_fingerprint"] != right["manifest_fingerprint"]:
        raise ValueError("Cannot compare different RoboLab manifests")
    if not left["complete"] or not right["complete"]:
        raise ValueError("Cannot compare incomplete RoboLab evaluations")
    if not left.get("provenance_verified") or not right.get("provenance_verified"):
        raise ValueError("Comparison requires verified policy and simulator provenance for every instruction level")
    left_versions = {json.dumps(r["simulator_versions"], sort_keys=True) for r in left["run_receipts"].values()}
    right_versions = {json.dumps(r["simulator_versions"], sort_keys=True) for r in right["run_receipts"].values()}
    if len(left_versions) != 1 or left_versions != right_versions:
        raise ValueError("Cannot compare different simulator versions")
    checkpoints = {
        receipt["server_metadata"]["checkpoint_fingerprint"]
        for summary in (left, right)
        for receipt in summary["run_receipts"].values()
    }
    if len(checkpoints) != 1:
        raise ValueError("Cannot compare different checkpoint contents")

    def indexed(summary):
        return {(r["instruction_type"], r["task"], r["run"], r["env_id"]): r["success"] for r in summary["episodes"]}

    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise ValueError("Comparison requires exactly matching episode identities")
    return {
        "manifest_fingerprint": left["manifest_fingerprint"],
        "model": left["model"],
        "model_revision": left["model_revision"],
        "checkpoint_fingerprint": next(iter(checkpoints)),
        "simulator_versions": json.loads(next(iter(left_versions))),
        "left_backend": left["backend"],
        "right_backend": right["backend"],
        "paired_episodes": len(a),
        "both_success": sum(a[key] and b[key] for key in a),
        "left_only_success": sum(a[key] and not b[key] for key in a),
        "right_only_success": sum(not a[key] and b[key] for key in a),
        "both_failure": sum(not a[key] and not b[key] for key in a),
        "by_instruction": {
            level: {
                "left_success_rate": left["by_instruction"][level]["task_mean_success_rate"],
                "right_success_rate": right["by_instruction"][level]["task_mean_success_rate"],
                "delta_right_minus_left": right["by_instruction"][level]["task_mean_success_rate"]
                - left["by_instruction"][level]["task_mean_success_rate"],
            }
            for level in left["by_instruction"]
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Pin official checkouts, tasks and shared evaluation conditions")
    plan.add_argument("--robolab-root", type=Path, required=True)
    plan.add_argument("--cosmos-root", type=Path, required=True)
    plan.add_argument("--model", required=True)
    plan.add_argument("--model-revision", required=True)
    plan.add_argument("--tasks", nargs="+")
    plan.add_argument("--instruction-types", nargs="+", choices=INSTRUCTION_TYPES, default=list(INSTRUCTION_TYPES))
    plan.add_argument("--num-envs", type=int, default=10)
    plan.add_argument("--num-runs", type=int, default=1)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--policy-seed", type=int, default=0)
    plan.add_argument("--simulator-stack", choices=SIMULATOR_STACKS, default="isaac50")
    plan.add_argument("--manifest", type=Path, required=True)
    run = sub.add_parser("run", help="Run the official Cosmos3 client on the simulator host")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--robolab-root", type=Path, required=True)
    run.add_argument("--python", default=sys.executable, help="Python in RoboLab's Isaac environment")
    run.add_argument("--remote-uri", required=True)
    run.add_argument(
        "--backend", choices=("cuda", "npu", "naive"), required=True, help="Policy backend, not simulator device"
    )
    run.add_argument("--run-name", required=True)
    run.add_argument("--video-mode", choices=("all", "sensor", "viewport", "none"), default="all")
    run.add_argument(
        "--external-server", action="store_true", help="Allow an official server without a harness handshake"
    )
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summarize", help="Validate and score official episode results")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--results-root", type=Path, required=True)
    summary.add_argument("--allow-partial", action="store_true")
    summary.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare", help="Compare complete GPU/NPU runs under the same manifest and simulator")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        kwargs = vars(args).copy()
        for key in ("command", "manifest"):
            kwargs.pop(key)
        manifest = build_manifest(**kwargs)
        write_json(manifest, args.manifest)
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "fingerprint": manifest["fingerprint"],
                    "tasks": len(manifest["tasks"]),
                }
            )
        )
    elif args.command == "run":
        run_evaluation(args, load_manifest(args.manifest))
    elif args.command == "summarize":
        result = summarize_results(load_manifest(args.manifest), args.results_root, allow_partial=args.allow_partial)
        write_json(result, args.output)
        print(json.dumps({"complete": result["complete"], "output": str(args.output)}))
    else:
        write_json(compare_results(json.loads(args.left.read_text()), json.loads(args.right.read_text())), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
