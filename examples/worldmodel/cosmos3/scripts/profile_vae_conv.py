"""A/B the causal conv that lets the kernel pad H/W, alone and on top of the fused RMSNorm.

Arms: ``native`` (diffusers), ``fast_conv``, ``fast_conv+norm`` (the production path). The chunked window is held at
12 frames for every arm, so the only differences are inside the encoder. The gate is the usual one: the latent must
stay under the bf16 cast that ``packing.py`` applies anyway.

    python scripts/profile_vae_conv.py --checkpoint ... --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

RECIPE_CHUNK = 32
CHUNK_FRAMES = 12
ARMS = ("native", "fast_conv", "fast_conv+norm", "fast_enc+norm")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="DROID root. Without it the input is seeded noise.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--arms", default=",".join(ARMS))
    return parser.parse_args()


def build_vae(checkpoint: Path):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=torch.float32).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae


def real_batch(data, batch, max_episodes):
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
    return collate(chosen)["video"]


def encode_latent(vae, video):
    with torch.no_grad():
        return vae.encode(video).latent_dist.mode().float().cpu()


def main():
    import torch_npu

    args = parse_args()
    torch.npu.set_device(0)

    from vae_chunked import enable_chunked_encode
    from vae_fast_encoder import disable_fast_encoder, enable_fast_conv, enable_fast_encoder
    from vae_fused_norm import enable_fused_rmsnorm

    video = (
        (
            real_batch(args.data, args.batch, args.max_episodes)
            if args.data
            else torch.randn(args.batch, 3, RECIPE_CHUNK + 1, 544, 736)
        )
        .float()
        .to("npu")
    )
    vae = build_vae(args.checkpoint).to("npu:0")
    enable_chunked_encode(vae, args.chunk_frames)

    def configure(arm):
        disable_fast_encoder(vae)
        _reset_fused_norm(vae)
        if arm == "native":
            pass
        elif arm == "fast_conv":
            enable_fast_conv(vae)
        elif arm == "fast_conv+norm":
            enable_fast_conv(vae)
            enable_fused_rmsnorm(vae)
        elif arm == "fast_enc+norm":
            enable_fast_encoder(vae)
            enable_fused_rmsnorm(vae)
        else:
            raise SystemExit(f"unknown arm {arm}")

    reference = None
    print(
        f"{'dtype':>8} {'arm':>15} {'wall s':>8} {'vs base':>8} {'rel err':>10} {'vs bf16':>8} " f"{'peak GiB':>9}",
        flush=True,
    )

    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        vae.to(dtype)
        video_d = video.to(dtype)
        baseline_wall = None
        for arm in args.arms.split(","):
            configure(arm)
            with torch.no_grad():
                vae.encode(video_d)
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(args.iters):
                with torch.no_grad():
                    vae.encode(video_d)
            torch.npu.synchronize()
            wall = (time.perf_counter() - start) / args.iters
            peak = torch.npu.max_memory_allocated() / 2**30
            latent = encode_latent(vae, video_d)

            if reference is None:
                reference = latent
            delta = (reference - latent).abs()
            scale = reference.abs().mean().clamp_min(1e-8)
            cast = (reference - reference.bfloat16().float()).abs().mean()
            if baseline_wall is None:
                baseline_wall = wall
            print(
                f"{dtype_name:>8} {arm:>15} {wall:>8.3f} {baseline_wall / wall:>7.2f}x "
                f"{(delta.mean() / scale).item():>10.3e} {(delta.mean() / cast).item():>8.3f} {peak:>9.2f}",
                flush=True,
            )
        configure("native")


def _reset_fused_norm(vae):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

    for module in vae.modules():
        if isinstance(module, WanRMS_norm) and "forward" in module.__dict__:
            del module.__dict__["forward"]


if __name__ == "__main__":
    main()
