"""SFT of Cosmos3-Edge on DROID with ColossalAI on Ascend.

Follows the ``action_policy_droid_nano`` recipe of nvidia/cosmos-framework, with Edge's model config.

The checkpoint to pass is the **base**, ``nvidia/Cosmos3-Edge``. ``nvidia/Cosmos3-Edge-Policy-DROID`` is
that base with this very SFT already done to it -- measured tensor by tensor, 318 of its 549 transformer
tensors differ from the base and they are exactly the generation tower plus the action heads, while the 227
that match are the frozen understanding tower. It is a reference for where training should get to, not a
place to start from.

Launch with::

    python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
        --data /path/to/Cosmos3-DROID \
        --checkpoint /path/to/Cosmos3-Edge \
        --out /path/to/runs/cosmos3-sft

``torchrun`` from a virtualenv that inherits torch from the system interpreter may resolve to that interpreter
instead, which will not have ColossalAI importable; ``python -m torch.distributed.run`` avoids that.


The ``__main__`` guard is required: the DataLoader starts its workers with ``spawn``, which re-imports this
module in every worker, so any top-level side effect would run again there.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import torch
from droid_dataset import DroidActionPolicyDataset, collate
from packing import compute_loss, pack_batch

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
    parser.add_argument("--checkpoint", type=Path, required=True, help="Cosmos3-Edge directory (the base).")
    parser.add_argument("--out", type=Path, default=Path("runs/cosmos3-sft"), help="Checkpoint output directory.")
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
    parser.add_argument(
        "--reinit-action-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the action projections and the action modality embedding fresh instead of loading them. "
        "On by default, because the upstream recipe does it (checkpoint.keys_to_skip_loading) and because "
        "this script starts from a base checkpoint, whose action heads are not DROID-trained. Pass "
        "--no-reinit-action-head to keep the checkpoint's own, which only makes sense when continuing from "
        "a policy checkpoint.",
    )
    parser.add_argument(
        "--action-head-from",
        type=Path,
        help="Load the action projections and the modality embedding from a file instead of drawing them. "
        "For a comparison against the reference implementation: both sides re-initialize these from their "
        "own RNG, so one side has to export what it drew and the other load it, or the comparison starts "
        "from two different models. The file holds the reference implementation's own names (action2llm, "
        "llm2action, action_modality_embed), which are mapped here.",
    )
    parser.add_argument(
        "--cast-inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the plugin cast every floating-point forward argument to the training dtype. Off, because "
        "one of this model's arguments must not be cast -- see the comment at the plugin construction. It is "
        "a flag rather than a constant so that a reader can reproduce the difference instead of taking the "
        "number in the pull request on trust.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        default="full",
        choices=["full", "none"],
        help="Recompute transformer blocks in the backward instead of keeping their activations. 'full' is the "
        "default because without it the backward does not fit: one sample per rank on four cards fails in the "
        "attention backward kernel. 'none' exists to measure the cost.",
    )
    args = parser.parse_args()
    if args.data is None:
        parser.error("--data is required")
    return args


def reinitialize_action_head(model) -> list[str]:
    """Draw the action projections and the action modality embedding fresh, as the recipe does.

    ``action_policy_droid_nano`` lists them in ``checkpoint.keys_to_skip_loading``::

        # Skip net_ema. (EMA warm-starts from net, see dcp.py) and the action
        # heads, so they init fresh from the base (the base has no DROID-trained
        # action heads).
        keys_to_skip_loading=["net_ema.", "action2llm", "llm2action",
                              "action_modality_embed", "action_pos_embed"]

    so upstream's run starts the action head from its initializer even though the base checkpoint carries
    action weights. The initializer is ``DomainAwareLinear.initialize_action_parameters``: the per-domain
    table is ``trunc_normal_(std=1/sqrt(input_size))`` truncated at three sigma, the per-domain bias is zero,
    and the modality embedding uses ``std=1/sqrt(hidden_size)``.

    Measured on (episode 26827, offset 0) at a pinned sigma of 0.5 with the noise pinned on both sides, this
    is worth 1.149 against upstream's 1.883 in the action term of the loss -- the whole of a 39% gap that the
    inputs could not explain, since the clean action is bit-identical and the vision term agrees to 0.28%.

    It is on by default, because this script starts from the base checkpoint, as the recipe does. It stays a
    flag rather than the only behaviour because continuing from a policy checkpoint -- ``Cosmos3-Edge-Policy-
    DROID``, say -- is the one case where the released heads are the thing worth keeping.

    Returns:
        list[str]: Which modules were drawn again, so a caller can report it or fail on an empty list.
    """
    import math

    touched = []
    for name in ("action_proj_in", "action_proj_out"):
        module = getattr(model, name, None)
        if module is None:
            continue
        std = 1.0 / math.sqrt(module.input_size)
        torch.nn.init.trunc_normal_(module.fc.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(module.bias.weight)
        touched.append(name)
    embed = getattr(model, "action_modality_embed", None)
    if embed is not None:
        std = 1.0 / math.sqrt(embed.shape[-1])
        torch.nn.init.trunc_normal_(embed, std=std, a=-3 * std, b=3 * std)
        touched.append("action_modality_embed")
    return touched


def load_action_head(model, path: Path) -> list[str]:
    """Overwrite the action head with one drawn elsewhere, so two runs start from the same weights.

    The file is a mapping of the reference implementation's parameter names to tensors, as its own run
    exported them: ``action2llm.fc.weight`` / ``action2llm.bias.weight`` (this model's ``action_proj_in``),
    ``llm2action.*`` (``action_proj_out``) and ``action_modality_embed``. Anything else in the file is an
    error rather than something to ignore: a silently skipped tensor is a silently different model.

    Args:
        model (torch.nn.Module): The transformer.
        path (Path): The exported file, loaded with ``torch.load``.

    Returns:
        list[str]: The parameters that were overwritten.

    Raises:
        KeyError: If the file holds a name this model has no place for.
    """
    blob = torch.load(path, weights_only=False)
    # Two shapes of file. The whole export is every domain's row; the slice is one domain's, which is all a
    # single-embodiment comparison reads and 1/65th of the bytes to ship between two boxes. The slice is
    # named in the file rather than inferred, and what it leaves untouched is printed, because "the other
    # domains keep this run's own draw" is true but only safe while every sample is the named domain.
    domain = None
    if blob.get("format") == "action-head-domain-slice":
        domain = int(blob["domain_id"])
        blob = {k: v for k, v in blob.items() if k not in ("format", "domain_id")}
    loaded = []
    for key, value in blob.items():
        if key == "action_modality_embed":
            target = model.action_modality_embed
            row = target
        else:
            head, rest = key.split(".", 1)
            owner = {"action2llm": "action_proj_in", "llm2action": "action_proj_out"}.get(head)
            if owner is None:
                raise KeyError(f"{path}: {key!r} is not an action-head parameter this model has")
            target = model.get_parameter(f"{owner}.{rest}")
            row = target if domain is None else target[domain]
        if tuple(row.shape) != tuple(value.shape):
            raise ValueError(
                f"{path}: {key!r} is {tuple(value.shape)} but this model wants {tuple(row.shape)}"
                + ("" if domain is None else f" for domain {domain}")
            )
        with torch.no_grad():
            row.copy_(value.to(device=target.device, dtype=target.dtype))
        loaded.append(key)
    if domain is not None:
        print(
            f"[model] action head: only domain {domain} was replaced; the other domains keep this run's "
            "own draw and are read by no sample of that embodiment",
            flush=True,
        )
    return loaded


def build_model(checkpoint: Path, reinit_action_head: bool = False, action_head_from: Path = None):
    """Load the transformer and freeze the AR tower. The model comes from diffusers, not HF transformers."""
    from diffusers import Cosmos3OmniTransformer

    model = Cosmos3OmniTransformer.from_pretrained(checkpoint / "transformer", torch_dtype=torch.bfloat16)
    if reinit_action_head:
        print(f"[model] action head re-initialized: {', '.join(reinitialize_action_head(model))}", flush=True)
    if action_head_from is not None:
        # After the re-initialization on purpose: this replaces the draw rather than preceding it.
        print(
            f"[model] action head loaded from {action_head_from}: "
            f"{', '.join(load_action_head(model, action_head_from))}",
            flush=True,
        )
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
        # The parent directory holds both roots; the recipe's factory default trains the success root only.
        success_only=True,
    )
    collate_fn = collate
    coord.print_on_master(f"[data] {len(dataset)} window(s)")

    model = build_model(
        args.checkpoint,
        reinit_action_head=args.reinit_action_head,
        action_head_from=args.action_head_from,
    )
    model_config = model.config  # booster returns a wrapper, so keep the config before boosting
    if args.activation_checkpointing == "full":
        # The backward does not fit without this. At one sample per rank on four cards it failed inside
        # aclnnFlashAttentionUnpaddingScoreGrad, which is not a batch-size problem: the packed sequence is 3.6k
        # tokens over 36 blocks, and every block's inputs stay resident until its backward runs.
        #
        # This is the transformer's own switch, not the plugin's. HybridParallelPlugin exposes gradient
        # checkpointing for the models ShardFormer rewrites, and at tp=1 it rewrites nothing here, so there is
        # no module for it to wrap; the diffusers block already guards its own recompute on
        # ``self.gradient_checkpointing`` and reaches it through ``_gradient_checkpointing_func``. Calling the
        # diffusers API therefore reaches the blocks that matter and leaves ZeRO-2 untouched: what it changes is
        # when activations are produced, not how gradients are reduced or how optimizer state is sharded.
        #
        # Upstream recomputes blocks too, with torch's ``checkpoint_wrapper`` under
        # ``[model.activation_checkpointing] mode="full"``, plus ``save_ops_regex=["fmha"]``, which keeps the
        # fused-attention outputs so they are not recomputed (selective checkpointing). We recompute those as
        # well; that costs time, not correctness, and the selective variant is a later optimization.
        model.enable_gradient_checkpointing()
        if not getattr(model, "gradient_checkpointing", False):
            raise RuntimeError(
                "enable_gradient_checkpointing() left the flag off, so the blocks would not be recomputed and "
                "the backward would run out of memory instead of failing here."
            )
        coord.print_on_master(f"[model] activation checkpointing: {args.activation_checkpointing}")
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
        # The plugin otherwise casts every floating-point forward argument to bf16, and one of ours must not be
        # cast: `position_ids` is [3, sequence_length] of *fractional* mRoPE positions reaching 15163.8, and bf16
        # cannot hold a value that size closer than 16. Measured on one step, 3093 of 3244 positions come back
        # changed, off by as much as 17 whole positions, and the loss reads 5.492133 instead of 1.789169 -- the
        # rotary embedding is computed at the wrong place in the sequence. Peak memory is the same either way
        # (13.61 GiB): what stays fp32 is this index and the two timestep vectors.
        #
        # Nothing else needs casting by hand. The latents arrive bf16 from the VAE already, and the timesteps
        # are what the fp32 kept here is for: the transformer computes the sinusoidal projection from them,
        # in fp32 because they are fp32. Note what that line does next, though --
        # `self.time_embedder(self.time_proj(timesteps).to(time_embedder_dtype))` -- it rounds that fp32
        # projection to the embedder's parameter dtype before the call. Keeping the timesteps fp32 does not
        # prevent that; the policy does, by handing the embedder the value from before the cast.
        cast_inputs=args.cast_inputs,
        # And the cast must not reach the buffers either. This model's only floating-point buffer is
        # `rotary_emb.inv_freq`, the rotary frequency table: bf16 moves it by relL2 4.2e-04, which multiplied
        # by a position of 15163.8 is a phase error of radians rather than of last bits. Measured against the
        # reference implementation on one step with every input bit-identical, that one 64-element buffer put
        # the vision loss 0.1508% away from it instead of 0.0072% -- further than the reference's own bf16
        # against its own fp32 (0.101%) -- and leaving it alone makes this example's predictions
        # bit-identical to the same forward with no plugin at all.
        cast_buffers=False,
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
                coord.print_on_master(
                    f"[step {step}] loss {loss.item() * args.grad_accum:.6f} "
                    f"peak {torch.npu.max_memory_allocated() / 2**30:.2f} GiB"
                )
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
