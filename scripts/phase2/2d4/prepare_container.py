#!/usr/bin/env python3
"""Create a GPU-explicit validation container reusing the user's mounted checkout."""
import json
import os
import subprocess
from pathlib import Path

SOURCE = "wzj-colossalai-dev-tf517"
TARGET = os.environ.get("D2D4_CONTAINER", "wzj-colossalai-2d4")


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    original = json.loads(run("docker", "inspect", SOURCE))[0]
    if (
        subprocess.run(["docker", "inspect", TARGET], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        == 0
    ):
        raise RuntimeError(f"{TARGET} already exists; inspect and reuse it, do not overwrite")
    selected = os.environ.get("D2D4_GPUS", "0,1").split(",")
    rows = [
        line.split(",")
        for line in run(
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
        ).splitlines()
    ]
    inventory = {row[0].strip(): [v.strip() for v in row[1:]] for row in rows}
    uuids = []
    for index in selected:
        uuid, memory, util = inventory[index]
        if int(memory) > 256 or int(util) != 0:
            raise RuntimeError(f"GPU {index} is occupied")
        uuids.append(uuid)
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        TARGET,
        "--user",
        original["Config"]["User"],
        "--gpus",
        '"device=' + ",".join(uuids) + '"',
        "--shm-size",
        "8g",
        "--cpus",
        "8",
        "--memory",
        "40g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--workdir",
        "/workspace/ColossalAI",
    ]
    for node in [*(f"/dev/nvidia{i}" for i in selected), "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]:
        if Path(node).exists():
            command += ["--device", node]
    for mount in original["Mounts"]:
        assert mount["Type"] == "bind"
        value = f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount["RW"]:
            value += ",readonly"
        command += ["--mount", value]
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "VIRTUAL_ENV",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_MODULES_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_EXTENSIONS_DIR",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "PYTHONPATH",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
    }
    for item in original["Config"].get("Env", []):
        if item.partition("=")[0] in allowed:
            command += ["-e", item]
    command += ["--entrypoint", "/bin/bash", original["Image"], "-lc", "exec sleep infinity"]
    print(
        json.dumps(
            {
                "name": TARGET,
                "image": original["Image"],
                "physical_gpu_indices": selected,
                "gpu_uuids": uuids,
                "checkout": "/mnt/beegfs/ColossalAI/wangzhijian/src/ColossalAI",
            }
        )
    )
    print(run(*command))


if __name__ == "__main__":
    main()
