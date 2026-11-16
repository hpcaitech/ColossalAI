"""Does pinning the dataloader to a socket make ``__getitem__`` faster?  Measure before restarting a run for it.

The 48-rank runs pay 4-5 s per step because some rank's batch is late and everyone else waits at the end-of-backward
collective; node5/node9 stand out with about twice the data wait of the other four machines.  The proposed fix is to
pin each rank's process tree to the socket its NPU sits on (``numa_affinity.py``), but restarting a training run to
find out costs a cold start plus the cache-warming transient.

This probe answers the question on one process instead: it replays the same ``__getitem__`` stages the loader runs
(read frames, augmentation, compose, canvas, quantise, resize+pad, normalise) over the same windows under different
affinity masks, and prints seconds per sample for each.  Run it on a machine that is not being used for training,
or with ``nice -n 19`` next to a live run.

    OMP_NUM_THREADS=2 nice -n 19 python scripts/probe_numa_data.py --windows 12 --rounds 2

The arms run in two passes and only the second pass is reported, so the inductor-compiled hue kernel and the page
cache are warm for every arm.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats
import time
from pathlib import Path

import torch

from examples.worldmodel.cosmos3.droid_dataset import (
    CAMS,
    DroidActionPolicyDataset,
    closest_canvas,
    compose_concat_view,
    resize_and_reflect_pad,
)
from examples.worldmodel.cosmos3.numa_affinity import parse_cpulist

DATA = "/mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success"
CACHE = "/mnt/disk0/jiangzs/nzf/action_cache"


def sample(dataset, args):
    """One window through the same stages as ``DroidActionPolicyDataset.__getitem__``; returns (seconds, stages)."""
    stages = {}
    started = time.perf_counter()
    cpu_started = time.process_time()
    for index in range(args.windows):
        win = dataset.windows[index]
        marks = {}
        step = time.perf_counter()
        views = {cam: dataset._read_frames(cam, win).float() / 255.0 for cam in CAMS}
        marks["read"] = time.perf_counter() - step
        step = time.perf_counter()
        views = dataset._augment(views)
        marks["augment"] = time.perf_counter() - step
        step = time.perf_counter()
        comp = compose_concat_view(views["wrist"], views["left"], views["right"])
        th, tw, _ = closest_canvas(*comp.shape[-2:], resolution=args.resolution)
        comp8 = (comp * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        video, _ = resize_and_reflect_pad(comp8.float(), th, tw, pad_columns=args.pad)
        video = video / 127.5 - 1.0
        marks["post"] = time.perf_counter() - step
        step = time.perf_counter()
        dataset._read_actions(win)
        marks["actions"] = time.perf_counter() - step
        for name, seconds in marks.items():
            stages[name] = stages.get(name, 0.0) + seconds
    return time.perf_counter() - started, time.process_time() - cpu_started, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-arms", type=int, default=2, help="how many of the node's sockets to test")
    parser.add_argument(
        "--arms",
        default="",
        help="semicolon-separated cpusets, first arm is the unpinned control; default: control + this node's "
        "NPU sockets from numa_affinity",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.arms:
        arms = [("unpinned", None)] + [
            (f"cpus {entry}", parse_cpulist(entry)) for entry in args.arms.split(";") if entry
        ]
    else:
        import subprocess

        from examples.worldmodel.cosmos3.numa_affinity import build_map, parse_device_buses

        buses = parse_device_buses(subprocess.run(["npu-smi", "info"], capture_output=True, text=True).stdout)
        seen, unique = set(), []
        for entry in build_map(buses):
            if entry not in seen:  # the 8 devices share 4 sockets (or 4 domains on node5/node9)
                seen.add(entry)
                unique.append((f"cpus {entry}", parse_cpulist(entry)))
        arms = [("unpinned", None)] + unique[: args.max_arms]

    dataset = DroidActionPolicyDataset(
        DATA,
        chunk_length=32,
        fps=15,
        resolution=str(args.resolution),
        augment=True,
        max_episodes=64,
        action_cache=Path(CACHE),
        video_content_pad=args.pad,
    )
    print(f"[numa] {len(dataset)} windows; arms: {', '.join(name for name, _ in arms)}", flush=True)

    baseline = sorted(os.sched_getaffinity(0))
    wall = {name: [] for name, _ in arms}
    cpu = {name: [] for name, _ in arms}
    stages_by_arm = {name: {} for name, _ in arms}
    try:
        for round_index in range(args.rounds):
            for name, cpus in arms:
                os.sched_setaffinity(0, cpus if cpus else baseline)
                seconds, cpu_seconds, stages = sample(dataset, args)
                wall[name].append(seconds / args.windows)
                cpu[name].append(cpu_seconds / args.windows)
                stages_by_arm[name] = stages
                print(
                    f"[numa] round {round_index + 1} {name:<14}"
                    f" {seconds / args.windows * 1e3:7.1f} ms wall  {cpu_seconds / args.windows * 1e3:7.1f} ms cpu",
                    flush=True,
                )
    finally:
        os.sched_setaffinity(0, baseline)

    # The machine is busy with training, so wall time mostly measures how much CPU the scheduler gave us.  CPU time
    # per sample is the arm's own cost and the number to compare; the wall/cpu ratio says how starved we were.
    use = slice(1, None) if args.rounds > 1 else slice(None)
    control_cpu = stats.median(cpu[arms[0][0]][use])
    print(f"\n[numa] median of rounds {2 if args.rounds > 1 else 1}..{args.rounds}, per sample")
    for name, _ in arms:
        arm_cpu = stats.median(cpu[name][use])
        arm_wall = stats.median(wall[name][use])
        stages = stages_by_arm[name]
        share = "  ".join(
            f"{k} {v / args.windows * 1e3:.0f}ms" for k, v in sorted(stages.items(), key=lambda kv: -kv[1])
        )
        print(
            f"  {name:<14} {arm_cpu * 1e3:7.1f} ms cpu  x{arm_cpu / control_cpu:5.3f} of control"
            f"  (wall {arm_wall * 1e3:6.1f} ms, {arm_wall / arm_cpu:4.1f}x)   {share}"
        )


if __name__ == "__main__":
    main()
