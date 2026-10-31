"""Run an allocated GPU batch in a disposable dependency-image container.

The caller must hold the host GPU lock until this function (including cleanup)
returns. Docker-client process cleanup alone does not stop a container.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import uuid
from pathlib import Path

OWNER_LABEL = "org.colossalai.ci.owner"
SCOPE_LABEL = "org.colossalai.ci.scope"
OWNER = "gpu-batches"
IMAGE_REF = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
SCOPE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}\Z")
IMAGE_INPUTS = (
    "Dockerfile",
    "Dockerfile.dockerignore",
    "requirements.lock",
    "requirements-torch.lock",
    "verify_image.py",
)


def check_image_recipe(source, image_inputs):
    for name in IMAGE_INPUTS:
        if (source / name).read_bytes() != (image_inputs / name).read_bytes():
            raise RuntimeError(f"CI image does not match {name}; build and qualify a new dependency image")


def docker(*args):
    try:
        return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Docker {args[0]} failed: {error.stderr.strip()}") from error


def resolve_image(reference):
    if not IMAGE_ID.fullmatch(reference) and not IMAGE_REF.fullmatch(reference):
        raise RuntimeError(
            "COLOSSAL_CI_IMAGE must be a full local sha256:<64 hex digits> image ID "
            "or an already-loaded registry image@sha256:digest"
        )
    # Inspect never pulls. Build/load the image on the runner's Docker daemon
    # before testing; no registry or upload is required for local image IDs.
    image_id = docker("image", "inspect", "--format", "{{.Id}}", reference)
    if not IMAGE_ID.fullmatch(image_id):
        raise RuntimeError("Docker returned an invalid image ID")
    if IMAGE_ID.fullmatch(reference) and image_id != reference:
        raise RuntimeError("Docker returned a different local image ID")
    return image_id


def validate_scope(scope):
    if not SCOPE.fullmatch(scope):
        raise RuntimeError("Invalid CI_CONTAINER_SCOPE")
    return scope


def create_command(name, scope, image_id, source, output, gpu_uuids, batch, env):
    validate_scope(scope)
    if not IMAGE_ID.fullmatch(image_id):
        raise RuntimeError("Container launch requires a resolved image ID")
    if (
        not gpu_uuids
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(not re.fullmatch(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", gpu) for gpu in gpu_uuids)
    ):
        raise RuntimeError("Container launch requires allocated GPU UUIDs")
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    if not (source / ".github/scripts/ci/run_container_batch.sh").is_file():
        raise RuntimeError("Container entry script is missing from the selected source")
    if any("," in str(path) or "\n" in str(path) for path in (source, output)):
        raise RuntimeError("Docker bind paths must not contain commas or newlines")

    cpus = env.get("COLOSSAL_CI_CPUS", "8")
    memory = env.get("COLOSSAL_CI_MEMORY", "64g")
    shm = env.get("COLOSSAL_CI_SHM", "8g")
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?", cpus):
        raise RuntimeError("COLOSSAL_CI_CPUS must be positive")
    if any(not re.fullmatch(r"[1-9][0-9]*[mg]", value) for value in (memory, shm)):
        raise RuntimeError("COLOSSAL_CI_MEMORY and COLOSSAL_CI_SHM must be positive m/g sizes")

    # No host venv, CUDA toolkit, home, Docker socket, credentials or compiler
    # cache is mounted. Reports are written as the runner UID/GID, not root.
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--pull=never",
        "--init",
        "--label",
        f"{OWNER_LABEL}={OWNER}",
        "--label",
        f"{SCOPE_LABEL}={scope}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cpus",
        cpus,
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--shm-size",
        shm,
        "--pids-limit=4096",
        "--ulimit=memlock=-1:-1",
        "--gpus",
        '"device=' + ",".join(gpu_uuids) + '"',
        "--mount",
        f"type=bind,src={source},dst=/ci-source,readonly",
        "--mount",
        f"type=bind,src={output},dst=/ci-results",
        "--workdir",
        "/tmp",
    ]
    container_env = {
        "NVIDIA_VISIBLE_DEVICES": ",".join(gpu_uuids),
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        # Do not reuse the host indices (e.g. 4,7) after device filtering.
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(len(gpu_uuids))),
        "FAST_MODE": env.get("FAST_MODE", "1"),
        "CI_SOURCE_SHA": env.get("CI_SOURCE_SHA", ""),
        "CI_IMAGE_ID": image_id,
        "GITHUB_RUN_ID": env.get("GITHUB_RUN_ID", "local"),
        "COLOSSAL_HF_OFFLINE": env.get("COLOSSAL_HF_OFFLINE", "1"),
        "MAX_JOBS": env.get("MAX_JOBS", "4"),
        "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST", "9.0"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in container_env.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend(
        [
            "--entrypoint",
            "/bin/bash",
            image_id,
            "/ci-source/.github/scripts/ci/run_container_batch.sh",
            batch,
            container_env["CUDA_VISIBLE_DEVICES"],
        ]
    )
    return command


def remove_owned_container(name, scope):
    """Remove only an existing container with this exact job's ownership labels."""
    # Listing first distinguishes 'absent' from a daemon/permission failure.
    ids = docker(
        "ps",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        f"label={OWNER_LABEL}={OWNER}",
        "--filter",
        f"label={SCOPE_LABEL}={validate_scope(scope)}",
        "--filter",
        f"name=^/{name}$",
    ).splitlines()
    for container_id in ids:
        info = json.loads(docker("inspect", container_id))[0]
        labels = info["Config"].get("Labels") or {}
        if info["Name"] != f"/{name}" or labels.get(OWNER_LABEL) != OWNER or labels.get(SCOPE_LABEL) != scope:
            raise RuntimeError("Refusing to remove a container with different ownership")
        docker("rm", "--force", container_id)


def run_container(batch, selected, source, output, image_id, env, stream, result):
    scope = validate_scope(env.get("CI_CONTAINER_SCOPE") or f"local-{uuid.uuid4().hex}")
    name = f"colossalai-ci-{scope}-{batch}-{uuid.uuid4().hex[:12]}"
    command = create_command(name, scope, image_id, source, output, [gpu for _, gpu in selected], batch, env)
    result.update({"container_name": name, "container_scope": scope, "image_id": image_id})

    def interrupted(signum, frame):
        raise RuntimeError(f"Container batch interrupted by signal {signum}")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        except subprocess.CalledProcessError as error:
            result["container_create_error"] = error.stderr
            raise
        client_code = stream(["docker", "start", "--attach", name], output / "pytest.log", env)
        state = json.loads(docker("inspect", "--format", "{{json .State}}", name))
        result["container_state"] = state
        if state.get("Running") or state.get("Status") != "exited":
            raise RuntimeError("Docker client exited before the container finished")
        if state.get("OOMKilled"):
            raise RuntimeError("Container exceeded its configured memory limit")
        if state.get("Error"):
            raise RuntimeError(f"Container runtime error: {state['Error']}")
        return state["ExitCode"] or client_code
    finally:
        # Also runs when create times out: the daemon might have created the
        # container despite losing the client's response. The job's always()
        # cleanup is a second safety net for runner cancellation.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            remove_owned_container(name, scope)
            result["container_cleanup"] = "completed"
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            result["container_cleanup"] = f"failed: {error}"
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def cleanup_scope(scope):
    """Job finalizer; never prune a daemon or remove another job's containers."""
    validate_scope(scope)
    ids = docker(
        "ps",
        "--all",
        "--quiet",
        "--no-trunc",
        "--filter",
        f"label={OWNER_LABEL}={OWNER}",
        "--filter",
        f"label={SCOPE_LABEL}={scope}",
    ).splitlines()
    for container_id in ids:
        info = json.loads(docker("inspect", container_id))[0]
        remove_owned_container(info["Name"].removeprefix("/"), scope)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cleanup-scope")
    mode.add_argument("--check-recipe", type=Path, help="Source docker/torch213 directory")
    parser.add_argument("--image-inputs", type=Path, default=Path("/opt/ci"))
    args = parser.parse_args()
    if args.cleanup_scope:
        cleanup_scope(args.cleanup_scope)
    else:
        check_image_recipe(args.check_recipe, args.image_inputs)
