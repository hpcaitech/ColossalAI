"""Re-sweep the chunk window under the production encoder: fused RMSNorm + the fast conv path.

The 12-frame knee in Sec. 11 was measured against diffusers' encoder, where every convolution paid for a materialised
H/W pad. Both of those changed since, so the window that minimises wall time has to be re-measured rather than
carried over. This script holds the encoder at its production configuration and sweeps the window width.

    python scripts/profile_vae_recipe.py --checkpoint ... --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

RECIPE_CHUNK = 32


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="DROID root. Without it the input is seeded noise.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--chunks", default="4,8,12,16,20,24")
    parser.add_argument("--max-episodes", type=int, default=2)
    return parser.parse_args()


def build_vae(checkpoint: Path, dtype):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval()
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


def main():
    import torch_npu  # noqa: F401

    args = parse_args()
    torch.npu.set_device(0)
    dtype = getattr(torch, args.dtype)

    from vae_chunked import enable_chunked_encode
    from vae_fast_encoder import enable_fast_encoder
    from vae_fused_norm import enable_fused_rmsnorm

    video = (real_batch(args.data, args.batch, args.max_episodes) if args.data else
             torch.randn(args.batch, 3, RECIPE_CHUNK + 1, 544, 736)).float().to("npu")
    vae = build_vae(args.checkpoint, dtype).to("npu:0")
    enable_fused_rmsnorm(vae)
    enable_fast_encoder(vae)

    # Reference: the fp32 encoder, whose loop is bit-identical to a 4-frame chunked one. Converting the module is
    # cheap and keeps this table comparable with the Sec. 11 sweep, which used the same reference.
    vae.to(torch.float32)
    enable_chunked_encode(vae, 4)
    with torch.no_grad():
        reference = vae.encode(video).latent_dist.mode().float().cpu()
    vae.to(dtype)
    video_d = video.to(dtype)

    baseline_wall = None
    print(f"{'chunk':>6} {'calls':>6} {'wall s':>8} {'vs 4':>6} {'rel err':>10} {'vs bf16':>8} {'peak GiB':>9}",
          flush=True)

    num_frame = video_d.shape[2]
    for chunk in [int(value) for value in args.chunks.split(",")]:
        enable_chunked_encode(vae, chunk)
        calls = 1 + -(-(num_frame - 1) // chunk)  # prime frame, then ceil((T-1)/chunk) windows
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
        with torch.no_grad():
            latent = vae.encode(video_d).latent_dist.mode().float().cpu()

        if baseline_wall is None:
            baseline_wall = wall
        delta = (reference - latent).abs()
        scale = reference.abs().mean().clamp_min(1e-8)
        cast = (reference - reference.bfloat16().float()).abs().mean()
        print(f"{chunk:>6} {calls:>6} {wall:>8.3f} {baseline_wall / wall:>5.2f}x {(delta.mean() / scale).item():>10.3e} "
              f"{(delta.mean() / cast).item():>8.3f} {peak:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
