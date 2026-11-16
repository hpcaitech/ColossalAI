"""What a half-precision VAE encode costs in accuracy, measured on real samples on all eight cards.

``profile_vae_deep.py --accuracy-pair`` answers this for one random input on one card. That is enough to rank the
dtypes, not enough to sign off on: the error depends on the content (a mostly-static wrist view and a fast-moving
shoulder view do not quantise alike), and a training step encodes a batch of them on every rank at once. So this
runs the same comparison over real dataset windows, spread across the whole node.

Every rank takes its own slice of the dataset and never talks to the others -- no process group, no HCCL. Each prints
one parseable line per dtype; the caller aggregates. That keeps the measurement free of the collective that has been
the flakiest part of this stack.

    torchrun --nproc_per_node=8 scripts/profile_vae_accuracy.py \
        --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
        --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success

The yardstick is the recipe's own quantisation: ``packing.py`` casts the latent to bfloat16 before the model sees it,
so an encode is free of cost when its error stays under that cast's error.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

RECIPE_CHUNK = 32  # frames per window minus one; the dataset yields 33 frames for an encode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4, help="Samples per encode; 4 matches the training recipe.")
    parser.add_argument("--batches", type=int, default=2, help="Encodes per rank.")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--checkpoint-mode", default="full", help="Only used to pick the augmentation-free path.")
    parser.add_argument("--verbose", action="store_true", help="One line per sample instead of per rank.")
    return parser.parse_args()


def build_vae(checkpoint: Path):
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=torch.float32).eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False
    return vae


def encode(vae, video):
    with torch.no_grad():
        return vae.encode(video).latent_dist.mode().float().cpu()


def main():
    import torch_npu

    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    int(os.environ.get("WORLD_SIZE", 1))
    torch.npu.set_device(local_rank)

    from droid_dataset import DroidActionPolicyDataset, collate

    dataset = DroidActionPolicyDataset(
        args.data,
        chunk_length=RECIPE_CHUNK,
        resolution="480",
        augment=False,  # a deterministic window: the same bytes go through every dtype
        max_episodes=args.max_episodes,
    )

    def batch_at(start):
        """A batch has to share one canvas to stack; DROID's three cameras always do, but do not assume it."""
        base = dataset[start % len(dataset)]
        canvas = base["canvas_hw"].tolist()  # a tensor: comparing tensors in `if` is ambiguous
        chosen, step = [base], 1
        while len(chosen) < args.batch:
            candidate = dataset[(start + step) % len(dataset)]
            if candidate["canvas_hw"].tolist() == canvas:
                chosen.append(candidate)
            step += 1
            if step > 4 * len(dataset):
                raise SystemExit("no single canvas covers a batch")
        return collate(chosen)["video"].to("npu", torch.float32)

    batches = [batch_at((rank * args.batches + index) * args.batch) for index in range(args.batches)]
    print(
        f"[rank {rank}] {len(dataset)} window(s), {len(batches)} batch(es) of {args.batch}, "
        f"canvas {tuple(batches[0].shape)}",
        flush=True,
    )

    vae = build_vae(args.checkpoint).to(f"npu:{local_rank}")
    reference = [encode(vae, video) for video in batches]
    cpu_state = {name: tensor.detach().to("cpu", torch.float32).clone() for name, tensor in vae.state_dict().items()}

    for name in args.dtypes.split(","):
        dtype = getattr(torch, name)
        # Set the parameter dtype first: load_state_dict copies into the existing parameters, so loading before the
        # cast would round fp32 -> fp16 -> bf16 for the second dtype instead of fp32 -> bf16.
        vae.to(dtype)
        vae.load_state_dict({key: value.to(dtype) for key, value in cpu_state.items()})
        rows = []
        for video, ref in zip(batches, reference):
            half = encode(vae, video.to(dtype))
            for sample in range(len(ref)):
                delta = (ref[sample] - half[sample]).abs()
                scale = ref[sample].abs().mean().clamp_min(1e-8)
                cast = (ref[sample] - ref[sample].bfloat16().float()).abs().mean()
                rows.append(
                    (
                        (delta.mean() / scale).item(),
                        delta.max().item(),
                        cast.item() / scale.item(),
                    )
                )
                if args.verbose:
                    print(
                        f"[rank {rank}] {name} relative {(delta.mean() / scale).item():.4e} "
                        f"max {delta.max().item():.4e}",
                        flush=True,
                    )
        relative = sorted(row[0] for row in rows)
        ratio = [row[0] / row[2] for row in rows]
        print(
            f"[rank {rank}] {name} n={len(rows)} relative median {relative[len(relative) // 2]:.4e} "
            f"min {relative[0]:.4e} max {relative[-1]:.4e} "
            f"maxabs {max(row[1] for row in rows):.4e} "
            f"ratio_vs_bf16_cast median {sorted(ratio)[len(ratio) // 2]:.3f} max {max(ratio):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
