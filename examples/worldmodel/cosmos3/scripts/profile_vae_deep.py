"""Where the VAE encoder's time actually goes, stage by stage and kernel by kernel.

The shallow profiler (``profile_vae.py``) answers "how long is one encode". This one answers "of that time, which
stages and which kernels", and writes both a Chrome/Perfetto trace and the Ascend profiler's per-kernel tables.

Two views, because they answer different questions:

* **stage** -- forward hooks on the encoder's own children (``conv_in``, each down block, ``mid_block``, ``norm_out``,
  ``conv_out``). Each hook synchronises on exit, so a stage's number is device wall time and not the CPU's dispatch
  time: without that, the hooks only measure how far the CPU ran ahead of the device, which on this workload is
  exactly backwards (the CPU is throttled, so the numbers look plausible and are still wrong).
* **kernel** -- ``torch_npu.profiler`` with the NPU activity. Plain ``torch.profiler`` silently records no device
  events on this stack, so its tables show only CPU ops; the Ascend profiler writes ``kernel_details.csv`` and
  ``op_statistic.csv`` per kernel next to the trace.

Run it on one card (the encoder is per-rank, so the other seven change nothing but the noise)::

    python scripts/profile_vae_deep.py --checkpoint ... --out /home/niuzifan/logs/vae-deep

``--channels-last`` converts the model and input to ``channels_last_3d`` before profiling, which is the first
candidate the optimisation plan has to test on this device.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RECIPE = (33, 544, 656)  # production canvas with --vae-content-pad 16 (see draft_cosmos3_online_opt_zh.md F3)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3, help="Encode calls to average the unhooked wall time over.")
    parser.add_argument("--channels-last", action="store_true", help="Convert model and input to channels_last_3d.")
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record Python call stacks in the operator table (for attributing unnamed transposes/copies).",
    )
    parser.add_argument(
        "--norm-kernel",
        action="store_true",
        help="Patch the fused channel-first RMSNorm kernel (vae_norm_kernel.py) before profiling.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Weight and activation dtype. fp32 is the training recipe; the lower ones measure what Cube math buys.",
    )
    parser.add_argument("--hf32", action="store_true", help="Leave torch.npu.conv.allow_hf32 on instead of off.")
    parser.add_argument("--save-latent", type=Path, help="Write the mode() latent here for a cross-run comparison.")
    parser.add_argument("--compare", type=Path, help="Latent of an earlier run; report this run's error against it.")
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run torch_npu.profiler. Off by default: the CANN parser spends minutes turning the data into csv.",
    )
    parser.add_argument(
        "--accuracy-pair",
        action="store_true",
        help="Time fp32, convert the model to --pair-dtype in place, time that, and report the latent difference.",
    )
    parser.add_argument(
        "--pair-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="The fast dtype --accuracy-pair converts to. fp16 keeps three more mantissa bits than bf16.",
    )
    return parser.parse_args()


def build_vae(checkpoint: Path, device, dtype=torch.float32, hf32=False):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = hf32
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae.to(device)


def stage_modules(encoder):
    """The encoder's own children, plus each down block by index: the order they run in."""
    modules = {}
    for name, module in encoder.named_children():
        if name == "down_blocks":
            for index, block in enumerate(module):
                modules[f"down_blocks.{index}"] = block
        else:
            modules[name] = module
    return modules


def attach_hooks(modules, sync):
    """Time every call of every stage; the encoder calls each one once per temporal chunk, so they accumulate."""
    records = {name: {"seconds": 0.0, "calls": 0} for name in modules}
    handles = []
    for name, module in modules.items():

        def start(mod, args, name=name):
            if sync:
                torch.npu.synchronize()
            records[name]["start"] = time.perf_counter()

        def end(mod, args, output, name=name):
            if sync:
                torch.npu.synchronize()
            records[name]["seconds"] += time.perf_counter() - records[name]["start"]
            records[name]["calls"] += 1

        handles.append(module.register_forward_pre_hook(start))
        handles.append(module.register_forward_hook(end))
    return records, handles


def main():
    import torch_npu  # registers torch.npu and the Ascend profiler; plain torch.profiler records no device events

    args = parse_args()
    frames, height, width = RECIPE
    torch.npu.set_device(0)
    dtype = torch.float32 if args.accuracy_pair else getattr(torch, args.dtype)  # the pair always starts from fp32
    vae = build_vae(args.checkpoint, "npu:0", dtype, args.hf32)
    from examples.worldmodel.cosmos3.vae_fast_encoder import enable_fast_encoder
    enable_fast_encoder(vae)  # the production path: H/W pad handed to the conv, dead clone dropped
    if args.norm_kernel:
        from examples.worldmodel.cosmos3.vae_norm_kernel import enable_kernel_rmsnorm
        enable_kernel_rmsnorm(vae)
    if args.channels_last:
        vae = vae.to(memory_format=torch.channels_last_3d)

    # Fixed seed: two runs of different dtypes or layouts are only comparable on the same input.
    torch.manual_seed(0)
    video = torch.randn(args.batch, 3, frames, height, width, dtype=torch.float32).to(dtype)
    if args.channels_last:
        video = video.to(memory_format=torch.channels_last_3d)
    video = video.to("npu")

    with torch.no_grad():
        vae.encode(video)
    torch.npu.synchronize()

    with torch.no_grad():
        reference = vae.encode(video).latent_dist.mode().float().cpu()
        torch.npu.synchronize()
        wall_start = time.perf_counter()
        for _ in range(args.iters):
            vae.encode(video)
        torch.npu.synchronize()
        wall = (time.perf_counter() - wall_start) / max(args.iters, 1)

    if args.accuracy_pair:
        # One process for both dtypes, so the two encodes see the same weights, the same input and the same host state:
        # the difference between them is then the change itself, not the run-to-run noise of loading from NFS.
        target = getattr(torch, args.pair_dtype)
        with torch.no_grad():
            vae = vae.to(dtype=target)
            video = video.to(target)
            vae.encode(video)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                vae.encode(video)
            torch.npu.synchronize()
            half_wall = (time.perf_counter() - start) / max(args.iters, 1)
            candidate = vae.encode(video).latent_dist.mode().float().cpu()
        delta = (reference - candidate).abs()
        cast = (reference - reference.bfloat16().float()).abs()
        scale = reference.abs().mean().item()
        print(f"[pair] fp32 {wall:.3f}s  {args.pair_dtype} {half_wall:.3f}s  speedup {wall / half_wall:.2f}x")
        print(f"[pair] difference max {delta.max().item():.4e} mean {delta.mean().item():.4e} "
              f"relative {delta.mean().item() / scale:.4e}")
        print(f"[pair] bf16 cast  max {cast.max().item():.4e} mean {cast.mean().item():.4e} "
              f"relative {cast.mean().item() / scale:.4e}")
        print(f"[pair] ratio of means {delta.mean().item() / cast.mean().item():.3f}")
        return

    records, handles = attach_hooks(stage_modules(vae.encoder), sync=True)
    with torch.no_grad():
        encode_start = time.perf_counter()
        vae.encode(video)
        torch.npu.synchronize()
        synced_wall = time.perf_counter() - encode_start
    for handle in handles:
        handle.remove()

    total = sum(record["seconds"] for record in records.values())
    print(f"[config] batch={args.batch} shape={frames}x{height}x{width} channels_last={args.channels_last} "
          f"dtype={args.dtype}")
    print(f"[stage] encode wall {wall:.3f}s; synced per-stage encode {synced_wall:.3f}s")
    print(f"[stage] encoder stages {total:.3f}s; other {synced_wall - total:.3f}s (patchify, quant_conv, cat)")
    rows = []
    for name, record in records.items():
        rows.append({"stage": name, "calls": record["calls"], "seconds": round(record["seconds"], 4),
                     "pct": round(100 * record["seconds"] / total, 1) if total else 0.0})
    for row in sorted(rows, key=lambda r: -r["seconds"]):
        print(f"[stage] {row['stage']:<18} calls={row['calls']:<5} {row['seconds']:>8.3f}s {row['pct']:>5.1f}%")

    tag = "-".join([("channels-last" if args.channels_last else "contiguous"), args.dtype,
                    "hf32" if args.hf32 else "fp32"])
    if args.save_latent:
        torch.save(reference, args.save_latent)
    if args.compare:
        # The latent is consumed as bfloat16 downstream (`packing.py` casts it before the model sees it), so the error
        # of casting the fp32 reference is the yardstick a faster encode has to stay under, not zero.
        other = torch.load(args.compare).float()
        delta = (reference - other).abs()
        bf16 = (reference - reference.bfloat16().float()).abs()
        scale = reference.abs().mean().clamp_min(1e-8)
        print(f"[error] vs {args.compare}: max {delta.max().item():.3e} mean {delta.mean().item():.3e} "
              f"relative {(delta.mean() / scale).item():.3e}")
        print(f"[error] bf16 cast of this run: max {bf16.max().item():.3e} mean {bf16.mean().item():.3e} "
              f"relative {(bf16.mean() / scale).item():.3e}")

    if not args.profile:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"stages-{tag}.json").write_text(json.dumps(
            {"wall": round(wall, 4), "synced_wall": round(synced_wall, 4), "rows": rows}, indent=1))
        return

    torch.npu.reset_peak_memory_stats()
    profiler = torch_npu.profiler
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
    )
    profile_dir = args.out / tag
    profile_dir.mkdir(parents=True, exist_ok=True)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        with_stack=args.with_stack,
        on_trace_ready=profiler.tensorboard_trace_handler(str(profile_dir)),
        profile_memory=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(2):
            with torch.no_grad():
                vae.encode(video)
            prof.step()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() / 2**30
    print(f"[op] peak {peak:.2f} GiB; ascend profile in {profile_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"stages-{tag}.json").write_text(json.dumps(
        {"wall": round(wall, 4), "synced_wall": round(synced_wall, 4), "peak_gib": round(peak, 2), "rows": rows},
        indent=1,
    ))


if __name__ == "__main__":
    main()
