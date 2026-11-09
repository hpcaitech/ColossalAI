"""Temporary, single-H20 validation of the unmodified Attention PR commit."""

import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

SHA = "9790b3bbda0fba8a3bb1933e6b3433d5c85ef02a"
IMAGE = "sha256:ae8ff34bf340a692803edca6e0f38a576fc1a2b206c01b4ef6f7c2a40dfd8c10"
GPU_INDEX = 2
GPU_UUID = "GPU-edc2987a-f7fa-07ec-47f6-939970a58a3c"
DEPS = Path("/mnt/beegfs/renjie/shared-ops-acceptance-20261103-b9fc2918")
PREVIOUS = Path("/mnt/beegfs/renjie/attention-acceptance-20261110")
HELPERS = {
    "acceptance_evidence.py": "5f0b9c15d86e90bdb047ae96ca265ea4ca280ae8968358804dd36a3c62489894",
    "gpu_process.py": "42fbbdc9bb92882706b2601e6dfb79054673cd9241388b919a6dd4d6c1fa6e15",
}
WHEELS = {
    "fa2": "c6f25ab31c81f0139d985fa8a434874717165c52f954270b564f87264b08d32c",
    "fa3": "67d81bb9b176ab63083e7614627de63a045f7ad7380465337fee699215d5f352",
}
AUDIT_HASH = "9af89cdf0afe6df91d8d6bee8170257a90064513e6af23bbe4e0feefbfff8da1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def output(*args):
    return subprocess.check_output(args, text=True, timeout=120).strip()


def main():
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    assert run_id.isdigit() and attempt.isdigit()
    assert socket.gethostname() == "gpu-h20-5"
    sys.path.insert(0, str(PREVIOUS / "source/.github/scripts/ci"))
    from container_batch import gpu_device_paths
    from run_gpu_batch import parse_pool, query_gpus, select_idle

    assert GPU_INDEX in (parse_pool(os.environ.get("COLOSSAL_GPU_POOL")) or set())
    ci_lock = Path(os.environ["CI_GPU_LOCK"])
    assert ci_lock.is_absolute() and ci_lock.parent.is_dir()
    locks = sorted({ci_lock, Path("/tmp/colossalai-ci-gpu.lock")})
    root = Path(f"/mnt/beegfs/renjie/attention-actions-{run_id}-{attempt}")
    root.mkdir(exist_ok=False)
    source, results = root / "source", root / "results"
    results.mkdir()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    published = Path("/mnt/beegfs/ColossalAI/shared-results/numerics/2F/attention") / (
        f"{stamp}-9790b3bb-gh{run_id}-{attempt}"
    )
    assert not published.exists()
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"results_dir={published}\n")
    name = f"renjie-attention-gh-{run_id}-{attempt}"
    state = {
        "source_commit": SHA,
        "workflow_commit": os.environ["GITHUB_SHA"],
        "actions_url": f"https://github.com/hpcaitech/ColossalAI/actions/runs/{run_id}",
        "runner": socket.gethostname(),
        "image": IMAGE,
        "gpu_index": GPU_INDEX,
        "gpu_uuid": GPU_UUID,
        "locks": [str(p) for p in locks],
        "status": "preflight",
        "runs": [],
        "started": time.time(),
    }

    def save():
        (results / "run.json").write_text(json.dumps(state, indent=2))

    def interrupted(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    shutil.copy2(__file__, results / "runner.py")
    try:
        for helper, expected in HELPERS.items():
            assert digest(DEPS / helper) == expected, helper
            shutil.copy2(DEPS / helper, results / helper)
        wheel_hashes = {}
        for backend, expected in WHEELS.items():
            wheels = list((DEPS / "wheels" / backend).glob("*.whl"))
            assert len(wheels) == 1, wheels
            assert digest(wheels[0]) == expected, backend
            wheel_hashes[str(wheels[0].relative_to(DEPS))] = expected
        assert output("docker", "image", "inspect", "--format", "{{.Id}}", IMAGE) == IMAGE
        output("git", "clone", "--no-hardlinks", "--no-checkout", str(PREVIOUS / "source"), str(source))
        output("git", "-C", str(source), "checkout", "--detach", SHA)
        assert output("git", "-C", str(source), "rev-parse", "HEAD") == SHA
        assert not output("git", "-C", str(source), "status", "--porcelain")
        manifest = {
            "commit": SHA,
            "dirty": False,
            "image": IMAGE,
            "workflow_commit": state["workflow_commit"],
            "wheel_hashes": wheel_hashes,
            "helpers": HELPERS,
            "files": {},
            "dependencies": {},
        }
        for filename in output("git", "-C", str(source), "ls-files").splitlines():
            if filename.startswith(
                (
                    "tests/kit/numerics/",
                    "tests/test_shardformer/test_layer/test_attention_backends.py",
                    "colossalai/shardformer/",
                    "extensions/pybind/flash_attention/",
                    "colossalai/kernel/kernel_loader.py",
                )
            ):
                manifest["files"][filename] = digest(source / filename)
        for filename in ("source-manifest.json", "fa2-build.json", "fa3-build.json"):
            path = DEPS / "results" / filename
            if path.exists():
                manifest["dependencies"][filename] = json.loads(path.read_text())
        (results / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
        for directory in ("native-home-cache", "torch-extensions", "inductor", "triton"):
            (root / "cache" / directory).mkdir(parents=True)
        with ExitStack() as stack:
            for path in locks:
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state["gpu_checks"] = []
            for probe in range(2):
                inventory, processes = query_gpus()
                assert select_idle(inventory, processes, 1, {GPU_INDEX}) == [(GPU_INDEX, GPU_UUID)]
                state["gpu_checks"].append({"inventory": inventory, "processes": processes})
                if probe == 0:
                    time.sleep(2)
            env = {
                "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "/deps/compat-diffusers036:/work/source:/deps:/work",
                "PATH": "/deps/venv/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TENSORNVME_BACKEND": "pthread",
                "OMP_NUM_THREADS": "4",
                "MAX_JOBS": "4",
                "XDG_CACHE_HOME": "/work/cache",
                "TORCH_EXTENSIONS_DIR": "/work/cache/torch-extensions",
                "TORCHINDUCTOR_CACHE_DIR": "/work/cache/inductor",
                "TRITON_CACHE_DIR": "/work/cache/triton",
                "PYTHONDONTWRITEBYTECODE": "1",
                "COLOSSALAI_FA3_WHEEL_PATH": "/deps/wheels/fa3/flash_attn_3-3.0.1-cp310-abi3-linux_x86_64.whl",
            }
            launch = [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--pull=never",
                "--init",
                "--label",
                "org.colossalai.owner=renjie-attention-9790b3bb",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--group-add",
                "2017",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--cpus=12",
                "--memory=96g",
                "--memory-swap=96g",
                "--shm-size=8g",
                "--pids-limit=4096",
                "--ulimit=memlock=-1:-1",
                "--gpus",
                f'"device={GPU_UUID}"',
                "--mount",
                f"type=bind,src={root},dst=/work",
                "--mount",
                f"type=bind,src={root}/cache/native-home-cache,dst=/.cache",
                "--mount",
                f"type=bind,src={DEPS},dst=/deps,readonly",
                "--workdir",
                "/work/source",
            ]
            for device in gpu_device_paths([GPU_UUID]):
                launch += ["--device", str(device)]
            for key, value in env.items():
                launch += ["-e", f"{key}={value}"]
            state["container"] = name
            try:
                output(*(launch + [IMAGE, "sleep", "infinity"]))
                state["status"] = "running"
                save()
                for backend in ("fa2", "fa3", "sdpa", "all"):
                    directory = results / backend
                    directory.mkdir()
                    dest = f"/work/results/{backend}"
                    selection = "gpu" if backend == "all" else f"gpu and {backend}"
                    command = [
                        "/deps/venv/bin/python",
                        "/deps/gpu_process.py",
                        "pytest",
                        "-vv",
                        "--tb=short",
                        "-p",
                        "acceptance_evidence",
                        "tests/test_shardformer/test_layer/test_attention_backends.py",
                        "-k",
                        selection,
                        "--acceptance-evidence-dir",
                        dest + "/numerics",
                        "--acceptance-source-manifest",
                        "/work/results/source-manifest.json",
                        "--junitxml=" + dest + "/pytest.xml",
                        "-o",
                        "junit_family=xunit1",
                    ]
                    entry = {"backend": backend, "command": command, "started": time.time()}
                    state["runs"].append(entry)
                    save()
                    with (directory / "run.log").open("w") as log:
                        result = subprocess.run(
                            [
                                "docker",
                                "exec",
                                "-e",
                                "COLOSSALAI_TEST_ATTENTION_BACKEND=" + backend,
                                "-e",
                                "COLOSSALAI_ATTENTION_TRACE_DIR=" + dest + "/profiler",
                                "-e",
                                "COLOSSALAI_PROCESS_PREFLIGHT=" + dest + "/process-preflight.json",
                                name,
                                *command,
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=900,
                        )
                    entry.update(exit_code=result.returncode, finished=time.time())
                    save()
                    print(f"{backend}: pytest exit {result.returncode}", flush=True)
                state["status"] = "passed" if all(e["exit_code"] == 0 for e in state["runs"]) else "failed"
            finally:
                cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=60)
                state["container_removed"] = cleanup.returncode == 0
                if not state["container_removed"]:
                    state["status"] = "failed"
                    state["cleanup_error"] = cleanup.stderr
                save()
            inventory, processes = query_gpus()
            state["gpu_after"] = {"inventory": inventory, "processes": processes}
        audit_path = PREVIOUS / "audit_gpu.py"
        assert digest(audit_path) == AUDIT_HASH
        shutil.copy2(audit_path, results / "audit_gpu.py")
        subprocess.run([sys.executable, str(audit_path), str(results)], check=True, timeout=180)
        audit = json.loads((results / "audit.json").read_text())
        assert not audit["integrity_errors"]
        for backend, count in {"fa2": 203, "fa3": 203, "sdpa": 216, "all": 622}.items():
            assert audit["runs"][backend]["tests"] == count
            assert audit["runs"][backend]["skipped"] == 0
    except Exception as error:
        state.update(status="failed", error=repr(error))
        print(repr(error), flush=True)
    finally:
        state["finished"] = time.time()
        save()
        hashes = {str(p.relative_to(results)): digest(p) for p in sorted(results.rglob("*")) if p.is_file()}
        (results / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))
        shutil.copytree(results, published)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(f"Source: `{SHA}`\n\nWorkflow: `{state['workflow_commit']}`\n\n")
            stream.write(f"Physical GPU: {GPU_INDEX} / `{GPU_UUID}`\n\nStatus: **{state['status']}**\n\n")
            stream.write(f"Full evidence (including tensor traces): `{published}`\n\n")
            if (results / "audit.json").exists():
                audit = json.loads((results / "audit.json").read_text())
                stream.write("| Run | Passed | Failed | Errors | Skipped |\n|---|---:|---:|---:|---:|\n")
                for backend, stats in audit["runs"].items():
                    stream.write(
                        f"| {backend} | {stats['passed']} | {stats['failures']} | {stats['errors']} | {stats['skipped']} |\n"
                    )
    print(json.dumps(state, indent=2))
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
