"""SFT for Cosmos3-Edge-Policy-DROID with ColossalAI on Ascend.

Follows the ``action_policy_droid_nano`` recipe of nvidia/cosmos-framework, with Edge's model config.

Launch with::

    python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
        --data /path/to/Cosmos3-DROID/success \
        --checkpoint /path/to/Cosmos3-Edge-Policy-DROID \
        --out /path/to/runs/cosmos3-sft

``torchrun`` from a virtualenv that inherits torch from the system interpreter may resolve to that interpreter
instead, which will not have ColossalAI importable; ``python -m torch.distributed.run`` avoids that.


The ``__main__`` guard is required: the DataLoader starts its workers with ``spawn``, which re-imports this
module in every worker, so any top-level side effect would run again there.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
from droid_dataset import DroidActionPolicyDataset, collate
from packing import compute_loss, pack_batch
from torch.utils.data import Dataset

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

# --- Recipe constants, from action_policy_droid_nano. Not exposed as flags on purpose. ------------------------
LR = 2.0e-04  # paired with a global batch of 8192
BETAS = (0.9, 0.99)
EPS = 1.0e-08  # the recipe notes that bf16 parameters with eps 1e-6 make the action loss diverge
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Train the generation tower and the action heads; the AR tower stays frozen. These are the diffusers names; the
# cosmos-framework "moe_gen" parameters appear in diffusers as the *_moe_gen modules plus the add_* attention
# projections.
TRAINABLE_KEYS = (
    "_moe_gen",  # mlp_moe_gen / input_layernorm_moe_gen / post_attention_layernorm_moe_gen
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "k_norm_und_for_gen",
    "time_embedder",
    "proj_in",
    "proj_out",  # vae2llm / llm2vae
    "action_proj_in",
    "action_proj_out",  # action2llm / llm2action
    "action_modality_embed",
)
# These get 5x the base learning rate (the recipe's lr_multipliers).
LR_MULTIPLIED = ("action_proj_in", "action_proj_out", "action_modality_embed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="DROID dataset root, in LeRobot v3 layout.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Cosmos3-Edge-Policy-DROID directory.")
    parser.add_argument("--out", type=Path, default=Path("runs/cosmos3-sft"), help="Checkpoint output directory.")
    parser.add_argument(
    )

    # Recipe values by default; overridable for smoke runs at a shorter window or a smaller batch.
    parser.add_argument("--chunk-length", type=int, default=32, help="Actions per window; frames are this plus 1.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--resolution", default="480")
    parser.add_argument(
        "--samples-per-rank",
        type=int,
        default=32,
        help="8 ranks x 32 samples x 32 accumulation steps = 8192, the global batch of the 256-rank recipe.",
    )
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-episodes", type=int, help="Index only the first N episodes; the full index holds millions of windows."
    )
    parser.add_argument(
        "--mp-context",
        default="spawn",
        help="Decoding video in the parent before a fork deadlocks, so spawn is the default.",
    )
    parser.add_argument(
        "--vae-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Precision of the frozen VAE. bfloat16 matches the upstream recipe; float32 is 2.5x slower and twice "
        "the memory, and exists so a numerical question can be re-asked against a reference that does not round.",
    )
    parser.add_argument(
        "--backend",
        default="npu",
        choices=["npu", "naive"],
        help="Two-way attention backend. 'naive' is the PyTorch reference: it materializes the full score matrix "
        "and runs out of memory at the recipe's sequence length, so it is for numerical comparison only.",
    )
    args = parser.parse_args()
    if args.data is None:
        parser.error("--data is required")
    return args


def build_model(checkpoint: Path):
    """Load the transformer and freeze the AR tower. The model comes from diffusers, not HF transformers."""
    from diffusers import Cosmos3OmniTransformer

    model = Cosmos3OmniTransformer.from_pretrained(checkpoint / "transformer", torch_dtype=torch.bfloat16)
    trainable = frozen = 0
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in TRAINABLE_KEYS)
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    print(f"[model] trainable {trainable / 1e9:.3f}B / frozen {frozen / 1e9:.3f}B", flush=True)
    return model


def build_tokenizer(checkpoint: Path):
    """The text tower's tokenizer ships inside the checkpoint next to the transformer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint / "text_tokenizer")


def build_vae(checkpoint: Path, dtype: torch.dtype = torch.bfloat16):
    """Load the VAE frozen. It runs inside the training loop and never enters the booster.

    bfloat16 is the default because it is what the recipe runs: upstream's ``WanVAE`` defaults to it and the DROID
    posttrain config sets ``precision = "bfloat16"``. Measured on one 910B card against a CPU fp32 reference, on a
    real [1,3,33,544,736] window:

        fp32   1230 ms   6.12 GiB   relL2 8.4e-07 vs CPU
        bf16    500 ms   3.12 GiB   relL2 7.8e-03 vs CPU

    The bf16 latent sits 7.8e-03 away from the fp32 one, which is 1.3% of the distance between two augmented draws
    of the same window (0.614) -- far below the variation the recipe injects on purpose. fp32 stays reachable
    because the argument above was only settleable by having a reference that does not round.

    HF32 has to be off whenever a convolution does run in fp32. With it on, the error of an fp32 convolution on
    Ascend was measured at 210x the CPU result, which leaves the latents subtly wrong; the symptom is a loss curve
    that does not match the reference device, which is very hard to trace back to here. The flag is set
    unconditionally so that switching dtype can never silently re-enable it.
    """
    from diffusers import AutoencoderKLWan

    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    # from_pretrained, not .to(): diffusers keeps dtype-sensitive submodules in fp32 when it loads the weights
    # itself, and a cast after the fact bypasses that bookkeeping.
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def build_optimizer(model):
    """FusedAdam with fp32 master weights; the action heads get 5x the learning rate."""
    base, boosted = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (boosted if any(k in name for k in LR_MULTIPLIED) else base).append(p)
    return HybridAdam(
        [{"params": base, "lr": LR}, {"params": boosted, "lr": LR * 5.0}],
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        adamw_mode=True,
    )


# ---------------------------------------------------------------------------------------------------------------
def main():
    args = parse_args()

    # torch_npu registers the 'npu' device and its operators as an import side effect; no name from it is used
    # here, so it goes through importlib rather than as an import statement nothing references.
    importlib.import_module("torch_npu")

    # launch_from_torch defaults to backend='nccl', which Ascend cannot use. Reading it off the accelerator rather
    # than hard-coding 'hccl' keeps the same script runnable on a GPU host.
    from colossalai.accelerator import get_accelerator

    colossalai.launch_from_torch(backend=get_accelerator().communication_backend)
    coord = DistCoordinator()

    dataset = DroidActionPolicyDataset(
        args.data,
        chunk_length=args.chunk_length,
        fps=args.fps,
        resolution=args.resolution,
        augment=True,
        max_episodes=args.max_episodes,
    )
    collate_fn = collate
    coord.print_on_master(f"[data] {len(dataset)} window(s)")

    model = build_model(args.checkpoint)
    model_config = model.config  # booster returns a wrapper, so keep the config before boosting
    tokenizer = build_tokenizer(args.checkpoint)
    vae = build_vae(args.checkpoint, getattr(torch, args.vae_dtype)).to(torch.npu.current_device())
    optimizer = build_optimizer(model)

    # Cosmos3OmniPolicy does two things: it installs the two-way attention processor in postprocess(), after the
    # ShardFormer module replacements so that nothing overwrites it, and with tp > 1 it applies the usual
    # column/row-parallel rewrites. At tp = 1 it rewrites no module and only installs the processor.
    #
    # custom_policy is only wired into HybridParallelPlugin -- LowLevelZero and Gemini have no such argument --
    # so the plugin has to be this one, with tp and pp at 1 and only its ZeRO in use.
    policy = Cosmos3OmniPolicy(backend=args.backend)

    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        zero_stage=2,
        precision="bf16",
        max_norm=GRAD_CLIP,
        custom_policy=policy,
    )
    booster = Booster(plugin=plugin)

    dataloader = plugin.prepare_dataloader(
        dataset,
        batch_size=args.samples_per_rank,
        shuffle=False,  # the shuffle lives in the index: episode order is shuffled, windows stay sequential
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        multiprocessing_context=args.mp_context if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"The dataloader yields no batch: {len(dataset)} sample(s), samples_per_rank="
            f"{args.samples_per_rank}, drop_last=True."
        )

    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

    model.train()
    step = 0
    for it, batch in enumerate(dataloader):
        # pack_batch installs the per-sample boundaries on the policy itself; see its docstring for why they are
        # not handed back to be installed here.
        inputs, targets = pack_batch(
            batch,
            vae,
            tokenizer,
            model_config,
            policy=policy,
            num_frames=args.chunk_length + 1,
            fps=args.fps,
            resolution=args.resolution,
            device=torch.npu.current_device(),
        )
        if it == 0:
            meta = policy.packed_meta
            coord.print_on_master(
                f"[pack] samples={meta.num_samples} AR={meta.total_ar_tokens} DM={meta.total_dm_tokens} "
                f"backend={policy.processor.backend}"
            )
        out = model(**inputs)
        loss = compute_loss(out, targets) / args.grad_accum
        booster.backward(loss, optimizer)

        if (it + 1) % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % args.log_every == 0 or step == 1:
                coord.print_on_master(f"[step {step}] loss {loss.item() * args.grad_accum:.6f}")
            if step % args.save_every == 0:
                save_dir = args.out / f"step-{step}"
                booster.save_model(model, str(save_dir), shard=True, use_safetensors=True)
                # safetensors' save_file bypasses the umask and writes 0600, so a checkpoint saved by one user is
                # unreadable to the rest of their group until the mode is fixed up.
                if coord.is_master():
                    for p in save_dir.rglob("*"):
                        os.chmod(p, 0o664 if p.is_file() else 0o775)
            if step >= args.max_iter:
                break

    coord.print_on_master("done")


if __name__ == "__main__":
    main()
