"""End-to-end check of the conditioning frames against the real model.

tests/test_target_form.py checks the same contracts against the packer alone. This one runs the actual
transformer, because the part that cannot be checked without it is that the prediction comes back **zero at the
conditioning frames** -- which is why the loss must apply the noisy mask rather than ignore it.

Needs an Ascend device, the checkpoint and the DROID dataset::

    python scripts/smoke_conditioning.py \
        --data /path/to/Cosmos3-DROID/success \
        --checkpoint /path/to/Cosmos3-Edge-Policy-DROID
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=4)
    return parser.parse_args()


class _Policy:
    """Stand-in for Cosmos3OmniPolicy: this script packs and runs the forward, it does not shard."""

    def __init__(self):
        self.packed_meta = None

    def set_packed_meta(self, meta):
        self.packed_meta = meta


def main():
    args = parse_args()

    # torch_npu registers the 'npu' device and its operators as an import side effect; no name from it is used
    # here, so it goes through importlib rather than as an import statement nothing references.
    importlib.import_module("torch_npu")
    import train_sft
    from diffusers import Cosmos3OmniTransformer
    from droid_dataset import DroidActionPolicyDataset, collate
    from packing import compute_loss, normalize_latent, pack_batch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    torch.npu.set_device(args.device)
    torch.npu.conv.allow_hf32 = False

    dataset = DroidActionPolicyDataset(args.data, max_episodes=args.max_episodes, sample_stride=32)
    loader = DataLoader(
        dataset,
        batch_size=args.samples,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        multiprocessing_context="spawn",
    )
    batch = next(iter(loader))

    print("\n== dataset ==", flush=True)
    check(
        "state is 8-dimensional (joint 7 + gripper 1), the action space",
        tuple(batch["state"].shape) == (args.samples, 8),
        str(tuple(batch["state"].shape)),
    )

    # Through build_vae, so this smoke test cannot drift away from the dtype and the HF32 setting training runs on.
    vae = train_sft.build_vae(args.checkpoint).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "text_tokenizer")
    model = Cosmos3OmniTransformer.from_pretrained(args.checkpoint / "transformer", torch_dtype=torch.bfloat16).to(
        args.device
    )
    model.eval()

    policy = _Policy()
    generator = torch.Generator().manual_seed(0)
    inputs, targets = pack_batch(
        batch,
        vae,
        tokenizer,
        model.config,
        policy=policy,
        num_frames=dataset.n_obs,
        fps=dataset.fps,
        resolution=dataset.resolution,
        device=args.device,
        generator=generator,
    )
    meta = policy.packed_meta
    t_latent, hp, wp = inputs["vision_token_shapes"][0]
    spatial = hp * wp
    n_act = inputs["action_token_shapes"][0][0]
    n = meta.num_samples

    print("\n== forward contract ==", flush=True)
    check(
        "vision noisy frames are 1..T-1, frame 0 conditioning",
        inputs["vision_noisy_frame_indexes"][0].tolist() == list(range(1, t_latent)),
        f"T={t_latent}",
    )
    check("the action stream carries one extra initial-state step", n_act == 33, f"n_act={n_act}")
    check(
        "action noisy steps are 1..n-1, step 0 conditioning",
        inputs["action_noisy_frame_indexes"][0].tolist() == list(range(1, n_act)),
    )
    want = n * (t_latent - 1) * spatial
    check(
        "vision_mse_loss_indexes covers only the noisy frames",
        inputs["vision_mse_loss_indexes"].numel() == want,
        f"{inputs['vision_mse_loss_indexes'].numel()} == {want}",
    )
    check("vision_timesteps is sized by noisy tokens", inputs["vision_timesteps"].numel() == want)
    check(
        "action_mse_loss_indexes covers only the noisy steps",
        inputs["action_mse_loss_indexes"].numel() == n * (n_act - 1),
    )
    check("action_timesteps is sized by noisy steps", inputs["action_timesteps"].numel() == n * (n_act - 1))
    check(
        "vision_sequence_indexes still covers every frame",
        inputs["vision_sequence_indexes"].numel() == n * t_latent * spatial,
    )

    print("\n== the conditioning frame really is clean ==", flush=True)
    with torch.no_grad():
        # Deliberately a second, independent encode rather than a reuse of pack_batch's: the check below is that
        # the two agree. It has to follow the VAE's dtype for the same reason pack_batch does.
        encoded = vae.encode(batch["video"].to(args.device, next(vae.parameters()).dtype))
        dist = getattr(encoded, "latent_dist", None)
        clean = dist.mode() if dist is not None else encoded[0]
        # Same scaling pack_batch applies, or this independent encode is not the same quantity and the
        # equality check below fails for a reason that has nothing to do with conditioning. diffusers returns
        # the raw latent and leaves latents_mean / latents_std to the caller; pack_batch applies them because
        # upstream does, inside its own encode.
        clean = normalize_latent(clean, vae).to(torch.bfloat16)
    fed = inputs["vision_tokens"][0]
    check("frame 0 handed to the model equals the clean latent", torch.equal(fed[0, :, 0], clean[0, :, 0]))
    check("frame 1 was noised (control)", not torch.equal(fed[0, :, 1], clean[0, :, 1]))
    check(
        "action step 0 equals the raw state",
        torch.equal(inputs["action_tokens"][0][0, :8].float().cpu(), batch["state"][0].to(torch.bfloat16).float()),
    )

    print("\n== forward and loss ==", flush=True)
    with torch.no_grad():
        out = model(**inputs)
    check("the prediction is full size", out.sample[0].shape[2] == t_latent, str(tuple(out.sample[0].shape)))
    check(
        "the conditioning frame's prediction is zero, so the loss has to mask it",
        out.sample[0][0, :, 0].abs().max().item() == 0.0,
    )
    check("a noisy frame's prediction is non-zero (control)", out.sample[0][0, :, 1].abs().max().item() > 0.0)
    check("the conditioning action step's prediction is zero", out.action[0][0].abs().max().item() == 0.0)

    loss = compute_loss(out, targets)
    check("the loss is finite", torch.isfinite(loss).item(), f"{loss.item():.6f}")

    masked = torch.stack(
        [
            ((p.float() - t.float()).pow(2) * m.float()).mean()
            for p, t, m in zip(out.sample, targets["vision"], targets["vision_noisy_mask"])
        ]
    ).mean()
    unmasked = torch.stack(
        [(p.float() - t.float()).pow(2).mean() for p, t in zip(out.sample, targets["vision"])]
    ).mean()
    print(
        f"\n  vision term: masked {masked.item():.6f} / unmasked {unmasked.item():.6f} "
        f"({unmasked.item() / masked.item():.2f}x inflation if the mask is dropped)",
        flush=True,
    )

    passed = sum(RESULTS)
    print(f"\n{'all passed' if all(RESULTS) else '*** FAILURES ***'}  {passed}/{len(RESULTS)}", flush=True)
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
