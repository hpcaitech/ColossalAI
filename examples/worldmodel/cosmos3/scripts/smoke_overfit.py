"""Overfit a single sample through the real training path.

What this shows: the optimization path is connected end to end -- gradients reach the parameters that are meant to
train, the loss is attached to the model output, the optimizer updates, and ZeRO-2 shards the gradients of a model
whose majority of parameters are frozen without corrupting them.

What it does **not** show: that the target is the right one. With 1.4B trainable parameters and one sample almost
any mapping is learnable, a flipped target sign included. That part is tests/test_target_form.py.

The one detail that makes or breaks the test is the **fixed sigma**. With sigma resampled per step the target
changes from step to step, the loss has a non-zero floor, and the curve says nothing. Everything else random is
removed by packing once and feeding the same tensors every step.

Needs an Ascend device, the checkpoint and the DROID dataset::

    python -m torch.distributed.run --nproc_per_node=1 scripts/smoke_overfit.py \
        --data /path/to/Cosmos3-DROID \
        --checkpoint /path/to/Cosmos3-Edge-Policy-DROID
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.5, help="Fixed noise level; see the module docstring.")
    parser.add_argument("--backend", default="npu", choices=["npu", "naive"])
    parser.add_argument("--min-ratio", type=float, default=10.0, help="Required ratio of the first loss to the last.")
    return parser.parse_args()


def main():
    args = parse_args()

    # torch_npu registers the 'npu' device and its operators as an import side effect; no name from it is used
    # here, so it goes through importlib rather than as an import statement nothing references.
    importlib.import_module("torch_npu")
    from torch.utils.data import DataLoader

    import colossalai
    from colossalai.accelerator import get_accelerator
    from colossalai.booster import Booster
    from colossalai.booster.plugin import HybridParallelPlugin
    from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

    colossalai.launch_from_torch(backend=get_accelerator().communication_backend)

    import packing
    import train_sft
    from droid_dataset import DroidActionPolicyDataset, collate

    device = torch.npu.current_device()

    # Remove every source of randomness the loop would otherwise reintroduce.
    packing.sample_sigma = lambda *a, **kw: args.sigma
    original_tokenize = packing.tokenize_prompt
    packing.tokenize_prompt = lambda prompt, tokenizer, **kw: original_tokenize(prompt, tokenizer, cfg_dropout_rate=0.0)

    dataset = DroidActionPolicyDataset(args.data, augment=False, max_episodes=2, success_only=True)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate, multiprocessing_context="spawn"
    )
    batch = next(iter(loader))
    print(f"[data] one sample: {batch['task'][0][:60]}", flush=True)

    model = train_sft.build_model(args.checkpoint)
    config = model.config
    tokenizer = train_sft.build_tokenizer(args.checkpoint)
    vae = train_sft.build_vae(args.checkpoint).to(device)
    optimizer = train_sft.build_optimizer(model)

    policy = Cosmos3OmniPolicy(backend=args.backend)
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        zero_stage=2,
        precision="bf16",
        max_norm=train_sft.GRAD_CLIP,
        custom_policy=policy,
    )
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    generator = torch.Generator().manual_seed(0)
    inputs, targets = packing.pack_batch(
        batch,
        vae,
        tokenizer,
        config,
        policy=policy,
        num_frames=dataset.n_obs,
        fps=dataset.fps,
        resolution=dataset.resolution,
        device=device,
        generator=generator,
    )
    meta = policy.packed_meta
    print(
        f"[pack] AR={meta.total_ar_tokens} DM={meta.total_dm_tokens} sigma={args.sigma} "
        f"t={inputs['vision_timesteps'][0].item():.1f}",
        flush=True,
    )
    del vae  # the loop never touches it again; give the memory back
    torch.npu.empty_cache()

    model.train()
    losses = []
    started = time.time()
    for step in range(1, args.steps + 1):
        out = model(**inputs)
        loss = packing.compute_loss(out, targets)
        booster.backward(loss, optimizer)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if step == 1 or step % args.log_every == 0:
            print(
                f"[step {step:3d}] loss {loss.item():10.6f}   ({(time.time() - started) / step:.2f}s/step)",
                flush=True,
            )

    first, last = losses[0], min(losses[-5:])
    ratio = first / max(last, 1e-12)
    print(f"\n[result] {first:.6f} -> {last:.6f}, a factor of {ratio:.1f}", flush=True)
    good = ratio >= args.min_ratio
    print(f"[{'PASS' if good else 'FAIL'}] the loss fell by at least {args.min_ratio:g}x", flush=True)
    if not good:
        print(
            "  A loss that will not move points at the packing or the target: the three position_ids axes, the "
            "order of the segments in the sequence, or where the noise is applied.",
            flush=True,
        )
    print(f"[peak_hbm_GB] {torch.npu.max_memory_allocated() / 2**30:.2f}", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
