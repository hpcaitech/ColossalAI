"""Measure VAE encoder optimizations at the DROID recipe shape in bfloat16.

Runs on one 910B2 card with the same real DROID batch through each arm. The core three
optimizations are chunked encoding (8 pixel frames), fused RMSNorm, and the fast encoder
(H/W conv padding plus removal of the residual-down-block clone). The fast encoder is
also split into its two changes so their individual effects are visible.

Example::

    python scripts/profile_vae_bf16_optimizations.py --checkpoint ... --data ...
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

RECIPE_CHUNK = 32
CHUNK_FRAMES = 8
ARMS = (
    "native",
    "chunk8",
    "fused_norm",
    "fast_conv",
    "fast_down",
    "fast_encoder",
    "all_three",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="DROID root")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iters", type=int, default=4, help="Timed encodes per measurement")
    parser.add_argument("--repeats", type=int, default=2, help="Measurements per arm, alternating sweep order")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--arms", default=",".join(ARMS))
    return parser.parse_args()


def build_vae(checkpoint: Path, dtype=torch.bfloat16):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae


def real_batch(data: Path, batch: int, max_episodes: int):
    from droid_dataset import DroidActionPolicyDataset, collate

    dataset = DroidActionPolicyDataset(
        data, chunk_length=RECIPE_CHUNK, resolution="480", augment=False, max_episodes=max_episodes
    )
    canvas = dataset[0]["canvas_hw"].tolist()
    chosen, step = [dataset[0]], 1
    while len(chosen) < batch:
        candidate = dataset[step % len(dataset)]
        if candidate["canvas_hw"].tolist() == canvas:
            chosen.append(candidate)
        step += 1
        if step > 4 * len(dataset):
            raise SystemExit("no single canvas covers a batch")
    video = collate(chosen)["video"]
    print(f"[data] {len(dataset)} window(s), canvas {tuple(video.shape)}", flush=True)
    return video


def reset_fused_norm(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    for module in vae.modules():
        if isinstance(module, WanRMS_norm) and "forward" in module.__dict__:
            del module.__dict__["forward"]


def configure(vae, native_encode, arm: str, chunk_frames: int):
    from vae_chunked import enable_chunked_encode
    from vae_fast_encoder import disable_fast_encoder, enable_fast_conv, enable_fast_downblock
    from vae_fused_norm import enable_fused_rmsnorm

    disable_fast_encoder(vae)
    reset_fused_norm(vae)
    vae._encode = native_encode

    if arm == "native":
        return
    if arm == "chunk8":
        enable_chunked_encode(vae, chunk_frames)
    elif arm == "fused_norm":
        enable_fused_rmsnorm(vae)
    elif arm == "fast_conv":
        enable_fast_conv(vae)
    elif arm == "fast_down":
        enable_fast_downblock(vae)
    elif arm == "fast_encoder":
        enable_fast_conv(vae)
        enable_fast_downblock(vae)
    elif arm == "all_three":
        enable_chunked_encode(vae, chunk_frames)
        enable_fused_rmsnorm(vae)
        enable_fast_conv(vae)
        enable_fast_downblock(vae)
    else:
        raise SystemExit(f"unknown arm: {arm}")


def encode_latent(vae, video):
    with torch.no_grad():
        return vae.encode(video).latent_dist.mode().float().cpu()


def fp32_reference(checkpoint: Path, video):
    vae = build_vae(checkpoint, torch.float32).to("npu:0")
    latent = encode_latent(vae, video.to("npu:0", torch.float32))
    del vae
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()
    return latent


def measure(vae, video, arm, native_encode, args):
    configure(vae, native_encode, arm, args.chunk_frames)
    with torch.no_grad():
        for _ in range(args.warmup):
            vae.encode(video)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            vae.encode(video)
    torch.npu.synchronize()
    wall = (time.perf_counter() - start) / args.iters
    peak = torch.npu.max_memory_allocated() / 2**30
    latent = encode_latent(vae, video)
    return wall, peak, latent


def main():
    import torch_npu

    args = parse_args()
    torch.npu.set_device(0)
    from vae_fast_encoder import disable_fast_encoder

    video_cpu = real_batch(args.data, args.batch, args.max_episodes)
    reference = fp32_reference(args.checkpoint, video_cpu)
    reference_scale = reference.abs().mean().clamp_min(1e-8)
    bf16_cast_error = (reference - reference.bfloat16().float()).abs().mean().clamp_min(1e-8)
    print(
        f"[fp32 reference] latent scale={reference_scale.item():.6e}, " f"bf16 cast error={bf16_cast_error.item():.6e}",
        flush=True,
    )
    video = video_cpu.to("npu:0", torch.bfloat16)
    del video_cpu
    vae = build_vae(args.checkpoint).to("npu:0")
    native_encode = vae._encode
    arms = [name.strip() for name in args.arms.split(",")]
    if args.repeats < 1 or args.iters < 1 or args.warmup < 0:
        raise SystemExit("repeats and iters must be positive; warmup cannot be negative")
    if not arms or arms[0] != "native":
        raise SystemExit("the first arm must be native so accuracy has a stable reference")
    schedule = []
    for sweep in range(args.repeats):
        schedule.extend(arms if sweep % 2 == 0 else reversed(arms))

    timings = {arm: [] for arm in arms}
    peaks = {arm: [] for arm in arms}
    errors = {arm: [] for arm in arms}
    bitwise = {}
    native_reference = None

    print(
        f"[setup] bf16, batch={args.batch}, frames={video.shape[2]}, canvas={video.shape[-2:]}, "
        f"chunk={args.chunk_frames}, iters={args.iters}, repeats={args.repeats}",
        flush=True,
    )
    print(
        f"{'arm':>14} {'rep':>4} {'wall s':>8} {'peak GiB':>9} {'rel native':>11} "
        f"{'rel fp32':>10} {'vs cast':>8} {'max native':>11} {'equal':>7}",
        flush=True,
    )

    for index, arm in enumerate(schedule):
        wall, peak, latent = measure(vae, video, arm, native_encode, args)
        if native_reference is None:
            native_reference = latent
        delta_native = (native_reference - latent).abs()
        rel_native = (delta_native.mean() / native_reference.abs().mean().clamp_min(1e-8)).item()
        delta_fp32 = (reference - latent).abs()
        rel_fp32 = (delta_fp32.mean() / reference_scale).item()
        ratio_vs_cast = (delta_fp32.mean() / bf16_cast_error).item()
        equal = torch.equal(native_reference, latent)
        timings[arm].append(wall)
        peaks[arm].append(peak)
        errors[arm].append((rel_native, rel_fp32, ratio_vs_cast, delta_native.max().item()))
        bitwise[arm] = bitwise.get(arm, True) and equal
        repeat = len(timings[arm])
        print(
            f"{arm:>14} {repeat:>4} {wall:>8.3f} {peak:>9.2f} {rel_native:>11.3e} "
            f"{rel_fp32:>10.3e} {ratio_vs_cast:>7.3f}x {delta_native.max().item():>11.3e} "
            f"{str(equal):>7}",
            flush=True,
        )

    baseline = statistics.median(timings["native"])
    print("\n[summary] median across ABBA measurements; errors vs native bf16 and fp32 reference", flush=True)
    print(
        f"{'arm':>14} {'median s':>9} {'speedup':>9} {'peak GiB':>9} {'rel native':>11} "
        f"{'rel fp32':>10} {'vs cast':>8} {'max native':>11} {'equal':>7}",
        flush=True,
    )
    for arm in arms:
        wall = statistics.median(timings[arm])
        rel_native = statistics.median(item[0] for item in errors[arm])
        rel_fp32 = statistics.median(item[1] for item in errors[arm])
        ratio_vs_cast = statistics.median(item[2] for item in errors[arm])
        max_abs = max(item[3] for item in errors[arm])
        peak = statistics.median(peaks[arm])
        print(
            f"{arm:>14} {wall:>9.3f} {baseline / wall:>8.2f}x {peak:>9.2f} {rel_native:>11.3e} "
            f"{rel_fp32:>10.3e} {ratio_vs_cast:>7.3f}x {max_abs:>11.3e} {str(bitwise[arm]):>7}",
            flush=True,
        )

    disable_fast_encoder(vae)
    reset_fused_norm(vae)
    vae._encode = native_encode


if __name__ == "__main__":
    main()
