"""How much the encoder-call window buys, and whether it changes the latent.

``AutoencoderKLWan._encode`` encodes one latent chunk (prime frame + 4 pixel frames) per encoder call. The
official tokenizer instead encodes 24 pixel frames per call at 480p and calls that the bulk of its tokenizer
speedup (report Sec. 5.2.6). ``vae_chunked.py`` installs that call pattern; this script checks the two things it
has to satisfy before it may go into training:

* **same latent** -- the causal caches are carried across calls, so widening the window should change nothing but
  floating-point summation order. The yardstick is again the bf16 cast that ``packing.py`` applies anyway.
* **cheaper** -- wall time per encode and peak HBM, for several window widths and both dtypes, on real DROID
  windows rather than noise.

Run it on one card (the encode is per-rank; the other seven change nothing but the noise)::

    python scripts/profile_vae_chunk.py --checkpoint ... --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success

``--chunks native,4,8,16,24,32`` sweeps the window; ``native`` leaves diffusers' own ``_encode`` in place and is
both the reference latent and the baseline time. Errors are quoted within a dtype, so the fp16 rows answer
"does the window change the latent" and not "does fp16 change the latent" -- that question has its own script.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

RECIPE_CHUNK = 32  # frames per dataset window minus one; the dataset yields 33 frames for an encode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="DROID root. Without it the input is seeded noise.")
    parser.add_argument("--batch", type=int, default=4, help="Samples per encode; 4 matches the training recipe.")
    parser.add_argument("--iters", type=int, default=5, help="Timed encode calls per setting.")
    parser.add_argument("--chunks", default="native,4,8,16,24,32")
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="torch.compile the encode before timing each setting.")
    parser.add_argument(
        "--compile-backend",
        default="inductor",
        help="torch.compile backend. inductor needs triton-ascend, which this image lacks; the "
        "native Ascend graph path is npugraph_ex (torchair).",
    )
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
    """One batch of real windows; all three views of an episode share a canvas, so stacking is safe."""
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


def latent_of(vae, video):
    """The latent the model actually consumes: the posterior mode, in fp32 on the host."""
    with torch.no_grad():
        return vae.encode(video).latent_dist.mode().float().cpu()


def main():
    import torch_npu

    args = parse_args()
    torch.npu.set_device(0)

    from vae_chunked import enable_chunked_encode

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
    native_encode = vae._encode  # bound method; restoring it undoes any patched window

    modes = [mode.strip() for mode in args.chunks.split(",")]
    dtypes = [name.strip() for name in args.dtypes.split(",")]
    fp32_reference = None

    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        vae.to(dtype)
        video_d = video.to(dtype)
        print(f"\n[dtype] {dtype_name}", flush=True)
        print(
            f"{'window':>8} {'calls':>6} {'wall s':>8} {'vs native':>10} {'rel err':>10} "
            f"{'vs bf16 cast':>13} {'peak GiB':>9} {'warm s':>8}",
            flush=True,
        )

        reference = None
        native_wall = None
        for mode in modes:
            calls = {"n": 0}
            counter = vae.encoder.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
            vae._encode = native_encode  # undoes a wider window from the previous row
            if mode != "native":
                enable_chunked_encode(vae, int(mode))
            if args.compile:
                vae._encode = torch.compile(vae._encode, backend=args.compile_backend, dynamic=False)

            try:
                with torch.no_grad():
                    warm_start = time.perf_counter()
                    vae.encode(video_d)  # warm-up: the first call pays any one-off allocation, or the compile
                    torch.npu.synchronize()
                warm = time.perf_counter() - warm_start
                torch.npu.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(args.iters):
                    with torch.no_grad():
                        vae.encode(video_d)
                torch.npu.synchronize()
                wall = (time.perf_counter() - start) / args.iters
                peak = torch.npu.max_memory_allocated() / 2**30
                latent = latent_of(vae, video_d)
            except RuntimeError as error:  # an OOM here bounds the window from above; a compile failure is a result too
                torch.npu.empty_cache()
                print(f"{mode:>8} {calls['n']:>6}  failed at batch {args.batch}: {str(error)[:200]}", flush=True)
                continue
            finally:
                counter.remove()

            if reference is None:
                reference = latent
                if fp32_reference is None:
                    fp32_reference = latent
                native_wall = wall
                speedup, rel, ratio = "-", "-", "-"
            else:
                delta = (reference - latent).abs()
                scale = reference.abs().mean().clamp_min(1e-8)
                cast = (reference - reference.bfloat16().float()).abs().mean()
                speedup = f"{native_wall / wall:.2f}x" if native_wall else "-"
                rel = f"{(delta.mean() / scale).item():.3e}"
                ratio = f"{(delta.mean() / cast).item():.3f}"
            print(
                f"{mode:>8} {calls['n']:>6} {wall:>8.3f} {speedup:>10} {rel:>10} {ratio:>13} "
                f"{peak:>9.2f} {warm:>8.1f}",
                flush=True,
            )

        if dtype_name != dtypes[0]:
            delta = (fp32_reference - reference).abs()
            scale = fp32_reference.abs().mean().clamp_min(1e-8)
            cast = (fp32_reference - fp32_reference.bfloat16().float()).abs().mean()
            print(
                f"[note] {dtype_name} native vs fp32 native: relative {(delta.mean() / scale).item():.3e}, "
                f"{delta.mean().item() / cast.item():.3f}x the bf16 cast",
                flush=True,
            )

    vae._encode = native_encode


if __name__ == "__main__":
    main()
