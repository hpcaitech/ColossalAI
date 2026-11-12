"""End-to-end check of the conditioning frames against the real model.

tests/test_target_form.py checks the same contracts against the packer alone. This one runs the actual
transformer, because the part that cannot be checked without it is that the prediction comes back **zero at the
conditioning frames** -- which is why the loss must apply the noisy mask rather than ignore it.

Needs a supported accelerator, the checkpoint and the DROID dataset::

    python scripts/smoke_conditioning.py \
        --data /path/to/Cosmos3-DROID/success \
        --checkpoint /path/to/Cosmos3-Edge-Policy-DROID
"""

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", help="Defaults to the current ColossalAI accelerator.")
    parser.add_argument("--backend", choices=("auto", "npu", "cuda", "naive"), default="auto")
    parser.add_argument("--vae-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=4)
    return parser.parse_args()


class _Policy:
    """Stand-in for Cosmos3OmniPolicy: this script packs and runs the forward, it does not shard."""

    def __init__(self, processor):
        self.packed_meta = None
        self.processor = processor

    def set_packed_meta(self, meta):
        self.packed_meta = meta
        self.processor.set_packed_meta(meta)


def main():
    args = parse_args()

    from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer
    from droid_dataset import DroidActionPolicyDataset, collate
    from packing import compute_loss, normalize_latent, pack_batch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from colossalai.accelerator import get_accelerator
    from colossalai.shardformer.modeling.cosmos3 import (
        ColossalTwoWayAttnProcessor,
        install_fp32_sinusoidal_timestep_input,
        install_fp32_timestep_embedding,
    )
    from examples.worldmodel.cosmos3.runtime import resolve_attention_backend, shutdown_dataloader

    accelerator = get_accelerator()
    args.device = args.device or str(accelerator.get_current_device())
    if args.device.startswith("npu"):
        importlib.import_module("torch_npu")
        torch.npu.set_device(args.device)
        torch.npu.conv.allow_hf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    backend = resolve_attention_backend(args.backend, torch.device(args.device).type)

    dataset = DroidActionPolicyDataset(args.data, max_episodes=args.max_episodes, sample_stride=32)
    loader = DataLoader(
        dataset,
        batch_size=args.samples,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        multiprocessing_context="spawn",
    )
    iterator = iter(loader)
    try:
        batch = next(iterator)
    finally:
        shutdown_dataloader(iterator)

    print("\n== dataset ==", flush=True)
    check(
        "state is 8-dimensional (joint 7 + gripper 1), the action space",
        tuple(batch["state"].shape) == (args.samples, 8),
        str(tuple(batch["state"].shape)),
    )

    vae_dtype = getattr(torch, args.vae_dtype)
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint / "vae", torch_dtype=vae_dtype).to(args.device)
    vae.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "text_tokenizer")
    model = Cosmos3OmniTransformer.from_pretrained(args.checkpoint / "transformer", torch_dtype=torch.bfloat16).to(
        args.device
    )
    model.eval()
    install_fp32_timestep_embedding(model)
    install_fp32_sinusoidal_timestep_input(model)
    processor = ColossalTwoWayAttnProcessor(backend=backend)
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ == "Cosmos3PackedMoTAttention":
            module.set_processor(processor)
            installed += 1
    if not installed:
        raise ValueError("Checkpoint has no Cosmos3 packed attention modules")

    policy = _Policy(processor)
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
        encoded = vae.encode(batch["video"].to(args.device, vae_dtype))
        distribution = getattr(encoded, "latent_dist", None)
        latent = distribution.mode() if distribution is not None else encoded[0]
        clean = normalize_latent(latent, vae).to(torch.bfloat16)
    fed = inputs["vision_tokens"][0]
    # The model receives normalized content cells, after removing the reflected canvas padding.
    clean = clean[:, :, :, : fed.shape[-2], : fed.shape[-1]]
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
