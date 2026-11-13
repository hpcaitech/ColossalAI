r"""Minimal ColossalAI training entry point for Cosmos 3.

The script is deliberately small: it owns the distributed launch, the plugin/policy wiring, metadata lifetime,
explicit sample/element loss objectives, checkpoint state and held-out evaluation. The data recipe (how a sample's latents,
targets, timesteps and loss masks are produced) stays outside; a replay pack is a ``.pt`` file holding a list of
packer-shaped sample dictionaries, exactly the input of
:func:`colossalai.shardformer.modeling.cosmos_mot.pack_cosmos3_samples`.

The real training path supports BF16 + DP/ZeRO and optional Ulysses context parallelism.
Tensor and pipeline parallelism are outside this entry point.

This is the harness half of the Cosmos 3 examples. The DROID action-policy SFT recipe lives next door in
``examples/worldmodel/cosmos3`` and owns the data pipeline (LeRobot decoding, the VAE, the prompt and the noise
schedule); this directory owns the batch contract, the benchmarks and the evaluation manifests, and runs on either
GPU or Ascend.

Run a CPU-only contract check (no diffusers, no distributed):

    python examples/worldmodel/cosmos3_harness/train.py --dry-run

Launch a real run on two H20 GPUs:

    torchrun --nproc_per_node=2 examples/worldmodel/cosmos3_harness/train.py \
        --model /path/to/Cosmos3-Edge --replay-pack /path/to/replay-packs.pt \
        --device cuda --attn-backend cuda --replay-mode replicated \
        --loss-objective generic-vision-token-mean --zero-stage 2 --steps 100 --save-dir /path/to/run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from colossalai.shardformer.modeling.cosmos3 import clear_cosmos3_packed_meta, set_cosmos3_packed_meta
from colossalai.shardformer.modeling.cosmos_mot import Cosmos3PackedBatch, pack_cosmos3_samples, two_way_varlen_summary

try:
    from examples.worldmodel.cosmos3_harness.heldout import load_split_manifest
    from examples.worldmodel.cosmos3_harness.loss import combine_modality_losses, normalise_packed_loss, velocity_mse
    from examples.worldmodel.cosmos3_harness.optim import build_adamw
    from examples.worldmodel.cosmos3_harness.training_metrics import (
        AnomalyDetector,
        ConsoleObserver,
        TrainingMetrics,
        WandbObserver,
        device_memory_gib,
    )
except ImportError:  # direct execution from the example directory
    from heldout import load_split_manifest
    from loss import combine_modality_losses, normalise_packed_loss, velocity_mse
    from optim import build_adamw
    from training_metrics import AnomalyDetector, ConsoleObserver, TrainingMetrics, WandbObserver, device_memory_gib


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Diffusers Cosmos 3 checkpoint directory or Hub id.")
    parser.add_argument("--replay-pack", help=".pt file with a list of packer-shaped sample dictionaries.")
    parser.add_argument("--eval-pack", help="Optional .pt file used as the fixed held-out pack.")
    parser.add_argument(
        "--heldout-manifest",
        help="JSON manifest written by heldout.py. Its fingerprint is verified, then the pack files are checked "
        "against its sample lists: a training sample in the validation list, or an eval sample outside it, is an "
        "error rather than a number quietly measured on a different split.",
    )
    parser.add_argument("--backend", "--attn-backend", default="auto", choices=("auto", "naive", "npu", "cuda"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "npu"), default="auto")
    parser.add_argument(
        "--replay-mode",
        choices=("replicated", "sharded"),
        required=False,
        help="replicated: one shared pack file, DP-strided; sharded: per-DP files, supports {dp_rank}.",
    )
    parser.add_argument(
        "--loss-objective",
        choices=("droid", "generic-vision-token-mean"),
        help="Required for real training; DROID replay samples carry complete 'targets' mappings.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument(
        "--checkpoint-mode", "--activation-checkpointing", choices=("none", "full", "selective"), default="full"
    )
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--zero-stage", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--adamw",
        default="auto",
        choices=("auto", "plain", "foreach", "fused"),
        help="AdamW implementation; the four arms share the same update math and exist for A/B measurement.",
    )
    parser.add_argument("--save-dir", help="Directory for model/optimizer checkpoints and training state.")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", help="Checkpoint directory written by this script.")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--bench-json",
        help="Write per-step timings, after the first --bench-warmup steps, to this JSON file; the report uses the "
        "same environment block and metric definitions as bench_ops.py.",
    )
    parser.add_argument("--bench-warmup", type=int, default=5, help="Steps excluded from --bench-json timings.")
    parser.add_argument(
        "--metrics-dir",
        help="Directory for the per-rank JSONL metrics (run.json, rank-*.jsonl, summary.json). Defaults to "
        "<save-dir>/metrics when --save-dir is set.",
    )
    parser.add_argument("--metrics-every", type=int, default=10, help="Aggregate and report metrics every N steps.")
    parser.add_argument("--wandb-project", help="Log the aggregated metrics to Weights & Biases from rank 0.")
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=("offline", "online", "disabled"),
        help="W&B mode. Offline is the default because training hosts are often isolated; sync the run directory "
        "later with 'wandb sync'.",
    )
    parser.add_argument("--wandb-run-name", help="W&B run name; defaults to W&B's generated name.")
    parser.add_argument("--dry-run", action="store_true", help="Exercise packing and loss on CPU; no model or GPU.")
    args = parser.parse_args(argv)
    for name in ("gradient_accumulation_steps", "cp_size", "save_interval", "log_interval", "metrics_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    return args


def resolve_backend(requested: str) -> str:
    """Resolve ``auto`` from the accelerator without hiding the choice in the policy default."""
    from colossalai.accelerator import get_accelerator
    from examples.worldmodel.cosmos3.runtime import resolve_attention_backend

    return resolve_attention_backend(requested, get_accelerator().name)


def _build_optimizer(
    model,
    learning_rate: float,
    backend: str = "auto",
    mode: str = "auto",
    *,
    device_type: Optional[str] = None,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    """Build the AdamW the training step uses; the arms and their trade-offs live in :mod:`optim`."""
    return build_adamw(
        model.parameters(),
        learning_rate,
        backend,
        mode,
        device_type=device_type,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def _metrics_dir(args: argparse.Namespace) -> Optional[str]:
    """Where the JSONL metrics go: --metrics-dir, else <save-dir>/metrics, else none."""
    if args.metrics_dir:
        return args.metrics_dir
    if args.save_dir:
        return str(Path(args.save_dir) / "metrics")
    return None


def _device_capacity_gib(device: torch.device) -> Optional[float]:
    """Total HBM of one device, for the memory-pressure check; None when the build cannot report it."""
    try:
        if device.type == "npu":
            pass

            return torch.npu.get_device_properties(device).total_memory / 2**30
        if device.type == "cuda":
            return torch.cuda.get_device_properties(device).total_memory / 2**30
    except Exception:
        return None
    return None


def _run_manifest(args: argparse.Namespace, device: torch.device, backend: str, world_size: int) -> Dict:
    """Identity of this run: configuration, topology and the environment block bench_ops already defines."""
    try:
        from examples.worldmodel.cosmos3_harness.bench_ops import collect_env
    except ImportError:  # direct execution from the example directory
        from bench_ops import collect_env

    return {
        "args": dict(vars(args)),
        "env": collect_env(device),
        "backend": backend,
        "world_size": world_size,
        "topology": {"tp": 1, "pp": 1, "cp": args.cp_size, "zero_stage": args.zero_stage},
        "model": args.model,
        "replay_pack": args.replay_pack,
        "eval_pack": args.eval_pack,
        "heldout_manifest": args.heldout_manifest,
    }


def _build_metrics(args: argparse.Namespace, rank: int, world_size: int, device: torch.device, backend: str):
    """Build the metrics facade, or None when no sink is configured."""
    out_dir = _metrics_dir(args)
    wandb_enabled = bool(args.wandb_project) and args.wandb_mode != "disabled"
    if out_dir is None and not wandb_enabled:
        return None

    aggregate_observers = [ConsoleObserver()]
    if wandb_enabled:
        aggregate_observers.append(WandbObserver(args.wandb_project, args.wandb_run_name, mode=args.wandb_mode))
    metrics = TrainingMetrics(
        out_dir=out_dir,
        rank=rank,
        world_size=world_size,
        log_every=args.metrics_every,
        aggregate_observers=aggregate_observers,
        detector=AnomalyDetector(device_capacity_gib=_device_capacity_gib(device)),
    )
    metrics.start(_run_manifest(args, device, backend, world_size))
    return metrics


def _synthetic_sample(und_len: int, dm_len: int, seed: int) -> Dict:
    """A packer-shaped sample with the fields the contract understands; used only by ``--dry-run``."""
    generator = torch.Generator().manual_seed(seed)
    sequence_length = und_len + dm_len
    return {
        "sample_id": f"synthetic-{seed}",
        "und_len": und_len,
        "sequence_length": sequence_length,
        "input_ids": torch.randint(0, 64, (und_len,), generator=generator),
        "text_indexes": torch.arange(und_len),
        "position_ids": torch.arange(sequence_length).unsqueeze(0).repeat(3, 1),
        "vision_sequence_indexes": torch.arange(und_len, sequence_length),
        "vision_mse_loss_indexes": torch.arange(und_len, sequence_length),
        "vision_timesteps": torch.full((dm_len,), 500.0),
        "vision_tokens": [torch.randn(1, 4, 2, 4, 4, generator=generator)],
        "vision_token_shapes": [(2, 2, 2)],
        "vision_noisy_frame_indexes": [torch.arange(1)],
        "vision_domain_ids": [torch.tensor(0)],
    }


def dry_run(args: argparse.Namespace) -> int:
    """Check the batch contract and the loss reduction without importing diffusers or initialising a process group."""
    samples = [_synthetic_sample(12, 48, seed=1), _synthetic_sample(8, 40, seed=2)]
    batch = pack_cosmos3_samples(samples, pad_tokens=4)
    prediction = torch.randn(batch.total_supervised_tokens)
    target = prediction + 0.01
    loss_sum, token_count = velocity_mse(prediction, target)
    combined = combine_modality_losses([{"sum": loss_sum, "count": token_count}])
    loss = normalise_packed_loss(combined["sum"], combined["count"], combined["count"], 1)

    report = {
        "status": "dry-run",
        "sample_ids": list(batch.sample_ids),
        "packed_tokens": batch.layout.sequence_length,
        "real_tokens": batch.layout.real_tokens,
        "pad_tokens": batch.layout.pad_tokens,
        "ar_dm_lengths": {"ar": batch.meta.ar_lens, "dm": batch.meta.dm_lens},
        "supervised_tokens": batch.total_supervised_tokens,
        "loss": float(loss.detach().cpu()),
        "varlen_summary": two_way_varlen_summary(batch.meta),
        "backend": resolve_backend(args.backend),
    }
    print(json.dumps(report, indent=2))
    if batch.total_supervised_tokens == 0:
        raise SystemExit("dry-run produced no supervised token")
    if not torch.isfinite(loss).item():
        raise SystemExit("dry-run produced a non-finite loss")
    return 0


def _load_replay_packs(path: str | Path) -> List[List[Dict]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must hold a non-empty list of packs (each pack is a list of sample dictionaries).")
    packs = [payload] if isinstance(payload[0], dict) else payload
    if any(
        not isinstance(pack, list) or not pack or any(not isinstance(sample, dict) for sample in pack) for pack in packs
    ):
        raise ValueError(f"{path} contains an empty or malformed replay pack")
    return packs


def _pack_sample_ids(packs: Sequence[Sequence[Dict]], source: str) -> set:
    """Collect the explicit sample ids of a pack file.

    The packer falls back to a sample's position when ``sample_id`` is missing, and a position is not comparable
    across pack files, so the held-out check requires the id to be written down.

    Args:
        packs: A pack file, as returned by :func:`_load_replay_packs`.
        source (str): Name of the file, for the error message.

    Returns:
        set: The sample ids, as strings.

    Raises:
        ValueError: If a sample carries no ``sample_id``.
    """
    sample_ids = set()
    for pack_index, pack in enumerate(packs):
        for sample_index, sample in enumerate(pack):
            if "sample_id" not in sample:
                raise ValueError(
                    f"{source}: pack {pack_index} sample {sample_index} carries no 'sample_id'. The held-out check "
                    f"matches samples by id; add the id to the pack rather than relying on its position."
                )
            sample_ids.add(str(sample["sample_id"]))
    return sample_ids


def _check_heldout_manifest(
    path: str | Path,
    training_packs: Sequence[Sequence[Dict]],
    eval_packs: Optional[Sequence[Sequence[Dict]]] = None,
) -> Dict:
    """Verify the split manifest and that the pack files stay on their side of it.

    :func:`load_split_manifest` proves the sample lists were not edited after the split was drawn. It cannot see a
    pack, so the two checks that need one live here: no training sample may be in ``validation_samples``, and every
    eval sample must be in it. Together they also make the eval and training packs disjoint.

    Args:
        path: Manifest written by :func:`heldout.write_split_manifest`.
        training_packs: Content of ``--replay-pack``.
        eval_packs (Optional): Content of ``--eval-pack``, or None when there is no held-out pack.

    Returns:
        Dict: The counts and the fingerprint, for the run log.

    Raises:
        ValueError: On an edited manifest, a pack sample without an id, training on held-out samples, or an eval
            sample the split never assigned to validation.
    """
    manifest = load_split_manifest(path)
    train_split = set(manifest["train_samples"])
    validation_split = set(manifest["validation_samples"])

    training_ids = _pack_sample_ids(training_packs, "the replay pack")
    leaked = sorted(training_ids & validation_split)
    if leaked:
        raise ValueError(
            f"The replay pack holds {len(leaked)} sample(s) the manifest assigns to validation, starting with "
            f"{leaked[:3]}. Training on them makes the held-out number a training number; rebuild the packs from "
            f"this manifest instead of editing either side."
        )

    summary = {
        "manifest": str(path),
        "fingerprint": manifest.get("fingerprint"),
        "train_split": len(train_split),
        "validation_split": len(validation_split),
        "training_samples": len(training_ids),
        "eval_samples": 0,
    }
    if eval_packs is None:
        return summary

    eval_ids = _pack_sample_ids(eval_packs, "the eval pack")
    outside = sorted(eval_ids - validation_split)
    if outside:
        raise ValueError(
            f"The eval pack holds {len(outside)} sample(s) outside the manifest's validation list, starting with "
            f"{outside[:3]}. The number would not be comparable with a run whose eval pack follows this manifest."
        )
    summary["eval_samples"] = len(eval_ids)
    return summary


def _to_device(batch: Cosmos3PackedBatch, device: torch.device) -> Cosmos3PackedBatch:
    moved = _tree_to_device(batch.model_inputs, device)
    return Cosmos3PackedBatch(
        model_inputs=moved,
        layout=batch.layout,
        sample_ids=batch.sample_ids,
        supervised_tokens=batch.supervised_tokens,
    )


def _tree_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _tree_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_to_device(v, device) for v in value)
    return value


def replay_window(packs, cursor, microsteps, dp_rank, dp_size, mode):
    """Cursor counts consumed pack slots (global for replicated, local for sharded).

    CP peers share dp_rank and therefore every pack in the window. Each window
    advances by all microsteps, including when wrapping the replay file.
    """
    if mode not in ("replicated", "sharded") or not packs or microsteps <= 0:
        raise ValueError("Invalid replay mode, empty packs, or nonpositive microsteps")
    stride = dp_size if mode == "replicated" else 1
    if mode == "replicated" and len(packs) < dp_size:
        raise ValueError("replicated replay needs at least one distinct pack per DP rank")
    offset = dp_rank if mode == "replicated" else 0
    window = [packs[(cursor + i * stride + offset) % len(packs)] for i in range(microsteps)]
    return window, cursor + microsteps * stride


def _pack_replay_samples(samples):
    """Targets are recipe metadata, never transformer inputs or packer tensor fields."""
    return pack_cosmos3_samples(
        [{key: value for key, value in sample.items() if key not in ("vision_target", "targets")} for sample in samples]
    )


def _sample_targets(sample):
    targets = sample.get("targets")
    required = {"vision", "action", "raw_action_dim", "vision_noisy_mask", "action_noisy_mask"}
    if not isinstance(targets, dict) or not required.issubset(targets):
        raise ValueError("DROID replay requires complete per-sample targets: " + ", ".join(sorted(required)))
    return targets


def _window_count(window, objective):
    if objective == "droid":
        for samples in window:
            for sample in samples:
                _sample_targets(sample)
        return sum(len(samples) for samples in window)
    if objective != "generic-vision-token-mean":
        raise ValueError("Choose an explicit loss objective")
    for samples in window:
        for sample in samples:
            if sample.get("action_tokens") or sample.get("sound_tokens"):
                raise ValueError("generic-vision-token-mean supports vision-only replay; choose droid for actions")
    return sum(sample["vision_target"].numel() for samples in window for sample in samples)


def _loss_for_batch(
    model,
    batch,
    samples,
    data_parallel_size,
    *,
    objective="generic-vision-token-mean",
    global_count=None,
    dp_group=None,
):
    """Return this microstep's contribution to the complete accumulation window."""
    device = next(model.parameters()).device
    set_cosmos3_packed_meta(model, batch.meta, strict=True)
    outputs = model(**_to_device(batch, device).model_inputs)
    if objective == "droid":
        from types import SimpleNamespace

        from examples.worldmodel.cosmos3.packing import compute_loss

        if (
            not outputs.sample
            or not outputs.action
            or len(outputs.sample) != len(samples)
            or len(outputs.action) != len(samples)
        ):
            raise ValueError("DROID output must have one vision and action prediction per replay sample")
        losses = []
        for i, sample in enumerate(samples):
            targets = _tree_to_device(_sample_targets(sample), device)
            # Call the recipe once per sample: no truncation, token weighting or dropped masks.
            targets = {k: v if k == "raw_action_dim" else [v] for k, v in targets.items()}
            losses.append(
                compute_loss(SimpleNamespace(sample=[outputs.sample[i]], action=[outputs.action[i]]), targets)
            )
        numerator, local_count = torch.stack(losses).sum(), len(samples)
    else:
        _window_count([samples], objective)
        combined = combine_modality_losses(
            [
                dict(zip(("sum", "count"), velocity_mse(outputs.sample[i], sample["vision_target"].to(device))))
                for i, sample in enumerate(samples)
            ]
        )
        numerator, local_count = combined["sum"], combined["count"]
    if global_count is None:
        global_count = _global_count(local_count, device, dp_group)
    return normalise_packed_loss(numerator, local_count, global_count, data_parallel_size)


def _global_count(count, device, group=None):
    value = torch.tensor(count, dtype=torch.int64, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, group=group)
    return int(value.item())


def _save_state(directory, model, optimizer, step, epoch, sampler_cursor, config=None):
    from examples.worldmodel.cosmos3_harness.checkpoint import save_checkpoint

    save_checkpoint(
        directory, model, optimizer, {"step": step, "epoch": epoch, "sampler_cursor": sampler_cursor, "config": config}
    )


def _restore_state(directory, model, optimizer, config=None):
    from examples.worldmodel.cosmos3_harness.checkpoint import restore_checkpoint

    return restore_checkpoint(directory, model, optimizer, expected_config=config)


def _write_bench_report(
    path: Path,
    args: argparse.Namespace,
    step_times: Sequence[float],
    world_size: int,
    backend: str,
    device: torch.device,
) -> None:
    """Write the step-time report; ``bench_ops.py`` owns the environment block so both reports are comparable."""
    timings = list(step_times)[args.bench_warmup :]
    if not timings:
        raise SystemExit("--bench-json needs more steps than --bench-warmup")
    try:
        from examples.worldmodel.cosmos3_harness.bench_ops import collect_env
    except ImportError:  # direct execution from the example directory
        from bench_ops import collect_env

    ordered = sorted(timings)
    report = {
        "mode": "train-step",
        "env": {**collect_env(device), "world_size": world_size, "backend": backend},
        "conditions": {
            "steps": len(timings),
            "warmup_steps": args.bench_warmup,
            "precision": args.precision,
            "zero_stage": args.zero_stage,
            "adamw": args.adamw,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "cp_size": args.cp_size,
            "loss_objective": args.loss_objective,
            "replay_mode": args.replay_mode,
            "timing_source": "device_synchronized_wall",
        },
        "step_ms": {
            "median": ordered[len(ordered) // 2] * 1e3,
            "min": ordered[0] * 1e3,
            "mean": sum(timings) / len(timings) * 1e3,
        },
        "steps_ms": [value * 1e3 for value in timings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    if not args.model or not args.replay_pack:
        raise SystemExit("a real run needs --model and --replay-pack; use --dry-run for the CPU contract check.")
    if not args.save_dir:
        raise SystemExit("a real run needs --save-dir so the checkpoint chain is explicit.")

    import torch.distributed as dist

    import colossalai
    from colossalai.accelerator import get_accelerator, set_accelerator
    from colossalai.booster import Booster
    from colossalai.booster.plugin import HybridParallelPlugin
    from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

    if args.device != "auto":
        set_accelerator(args.device)
    accelerator = get_accelerator()
    backend = resolve_backend(args.backend)
    if backend in ("cuda", "npu") and backend != accelerator.name:
        raise ValueError("Attention backend must match --device/the detected accelerator")
    if args.precision != "bf16" or accelerator.name not in ("cuda", "npu"):
        raise ValueError("Real harness training currently supports BF16 on CUDA/NPU only")
    if not args.loss_objective or not args.replay_mode:
        raise ValueError("Real training requires explicit --loss-objective and --replay-mode")
    import os

    if int(os.environ.get("WORLD_SIZE", "1")) % args.cp_size:
        raise ValueError("--cp-size must divide WORLD_SIZE")
    colossalai.launch_from_torch(backend=accelerator.communication_backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    backend = resolve_backend(args.backend)

    diffusers = __import__("diffusers")
    transformer_cls = getattr(diffusers, "Cosmos3OmniTransformer", None)
    if transformer_cls is None:
        raise SystemExit(
            f"diffusers {getattr(diffusers, '__version__', 'unknown')} does not expose Cosmos3OmniTransformer. "
            f"Install the pinned Cosmos 3 Diffusers source recorded in docs/HF5-COMPATIBILITY.md."
        )
    model = transformer_cls.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.train()
    if args.checkpoint_mode == "full" and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    policy = Cosmos3OmniPolicy(backend=backend, selective_checkpointing=args.checkpoint_mode == "selective")
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        sp_size=args.cp_size,
        enable_sequence_parallelism=args.cp_size > 1,
        sequence_parallelism_mode="all_to_all" if args.cp_size > 1 else None,
        cast_inputs=False,
        cast_buffers=False,
        precision=args.precision,
        zero_stage=args.zero_stage,
        custom_policy=policy,
        overlap_communication=True,
        max_norm=0.0,
    )
    optimizer = _build_optimizer(model, args.learning_rate, backend, args.adamw, device_type=accelerator.name)
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
    model._colossal_booster = booster

    metrics = _build_metrics(args, rank, world_size, next(model.parameters()).device, backend)

    dp_rank = dist.get_rank(plugin.dp_group)
    data_parallel_size = dist.get_world_size(plugin.dp_group)
    replay_path = args.replay_pack.format(dp_rank=dp_rank)
    packs = _load_replay_packs(replay_path)
    from examples.worldmodel.cosmos3_harness.checkpoint import file_digest

    fingerprints = [None] * world_size
    dist.all_gather_object(fingerprints, file_digest(replay_path))
    for dp in range(data_parallel_size):
        if len(set(fingerprints[dp * args.cp_size : (dp + 1) * args.cp_size])) != 1:
            raise ValueError("CP peers must read identical replay data")
    if args.replay_mode == "replicated" and len(set(fingerprints)) != 1:
        raise ValueError("replicated replay requires identical pack files on every rank")
    if args.replay_mode == "sharded" and len(set(fingerprints[:: args.cp_size])) != data_parallel_size:
        raise ValueError("sharded replay requires a distinct file per DP rank")
    config = {
        k: getattr(args, k)
        for k in (
            "loss_objective",
            "replay_mode",
            "cp_size",
            "precision",
            "zero_stage",
            "gradient_accumulation_steps",
            "checkpoint_mode",
            "model",
            "adamw",
            "learning_rate",
        )
    }
    config["attention_backend"] = backend
    config["replay_fingerprints"] = fingerprints
    config["heldout_fingerprint"] = file_digest(args.heldout_manifest) if args.heldout_manifest else None
    _window_count(packs, args.loss_objective)
    eval_packs = _load_replay_packs(args.eval_pack) if args.eval_pack else None
    eval_fingerprints = [None] * world_size
    dist.all_gather_object(eval_fingerprints, file_digest(args.eval_pack) if args.eval_pack else None)
    if len(set(eval_fingerprints)) != 1:
        raise ValueError("Every rank must use the identical held-out replay file")
    if eval_packs is not None:
        _window_count(eval_packs, args.loss_objective)
    if args.heldout_manifest:
        heldout = _check_heldout_manifest(args.heldout_manifest, packs, eval_packs)
        if rank == 0:
            print(json.dumps({"heldout": heldout}), flush=True)

    start_step = 0
    epoch = 0
    sampler_cursor = 0  # consumed global pack slots (replicated) or local pack slots (sharded)
    if args.resume:
        state = _restore_state(Path(args.resume), model, optimizer, config)
        start_step = int(state["step"])
        epoch = int(state["epoch"])
        sampler_cursor = int(state["sampler_cursor"])

    if start_step > args.steps:
        raise ValueError("--steps is smaller than the checkpoint's completed optimizer steps")
    count_data = args.cp_size == 1 or dist.get_rank(plugin.sp_group) == 0
    seen_samples = 0
    begin = time.time()
    step_times: List[float] = []
    timing_sources = set()
    last_saved_step = None
    for step in range(start_step, args.steps):
        measure_device = (
            bool(args.bench_json)
            or (step + 1) % args.log_interval == 0
            or (metrics is not None and (step + 1) % args.metrics_every == 0)
        )
        if measure_device:
            accelerator.synchronize()
        step_start = time.perf_counter()
        window, next_cursor = replay_window(
            packs, sampler_cursor, args.gradient_accumulation_steps, dp_rank, data_parallel_size, args.replay_mode
        )
        device = next(model.parameters()).device
        global_count = _global_count(_window_count(window, args.loss_objective), device, plugin.dp_group)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        batches = []
        for samples in window:
            batch = _pack_replay_samples(samples)
            batches.append(batch)
            try:
                micro_loss = _loss_for_batch(
                    model,
                    batch,
                    samples,
                    data_parallel_size,
                    objective=args.loss_objective,
                    global_count=global_count,
                    dp_group=plugin.dp_group,
                )
                booster.backward(micro_loss, optimizer)
                loss += micro_loss.detach()
            finally:
                clear_cosmos3_packed_meta(model, strict=True)
        optimizer.step()
        step_dispatch_ms = (time.perf_counter() - step_start) * 1e3
        if measure_device:
            accelerator.synchronize()
        step_ms = (time.perf_counter() - step_start) * 1e3
        step_times.append(step_ms / 1e3)
        timing_sources.add("device_synchronized_wall" if measure_device else "host_dispatch")
        sampler_cursor = next_cursor
        epoch = sampler_cursor // len(packs)
        # Report the actual window objective rather than rank zero's weighted contribution.
        dist.all_reduce(loss, group=plugin.dp_group)
        loss /= data_parallel_size
        sample_count = sum(len(samples) for samples in window)
        seen_samples += sample_count

        log_now = (step + 1) % args.log_interval == 0
        loss_value = float(loss.detach().cpu()) if log_now else None
        if rank == 0 and log_now and metrics is None:
            elapsed = time.time() - begin
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": loss_value,
                        "supervised_tokens": sum(b.total_supervised_tokens for b in batches),
                        "real_tokens": sum(b.layout.real_tokens for b in batches),
                        "pad_tokens": sum(b.layout.pad_tokens for b in batches),
                        "samples_per_second": seen_samples / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )
        if metrics is not None:
            sections = {
                "data": {
                    "samples": sample_count,
                    "supervised_tokens": sum(b.total_supervised_tokens for b in batches),
                    "ar_tokens": sum(b.meta.total_ar_tokens for b in batches),
                    "dm_tokens": sum(b.meta.total_dm_tokens for b in batches),
                    "real_tokens": sum(b.layout.real_tokens for b in batches),
                    "pad_tokens": sum(b.layout.pad_tokens for b in batches),
                },
                "perf": {
                    "step_ms": step_ms,
                    "step_dispatch_ms": step_dispatch_ms,
                    "timing_sources": {
                        "step_ms": "device_synchronized_wall" if measure_device else "host_dispatch",
                        "step_dispatch_ms": "host_dispatch",
                    },
                },
            }
            if not count_data:
                sections["data"] = {key: 0 for key in sections["data"]}
            if log_now:
                optim_section = {}
                getter = getattr(optimizer, "get_grad_norm", None)
                if getter is not None:
                    optim_section["grad_norm"] = getter()
                try:
                    optim_section["lr"] = optimizer.param_groups[0]["lr"]
                except (AttributeError, IndexError, KeyError):
                    optim_section["lr"] = args.learning_rate
                sections["loss"] = {"total": loss_value}
                sections["optim"] = optim_section
                sections["mem"] = device_memory_gib()
            metrics.step(step + 1, **sections)
        if args.save_dir and (step + 1) % args.save_interval == 0:
            _save_state(Path(args.save_dir), model, optimizer, step + 1, epoch, sampler_cursor, config)
            last_saved_step = step + 1

    if args.bench_json and rank == 0:
        _write_bench_report(
            Path(args.bench_json),
            args,
            step_times,
            world_size,
            backend,
            next(model.parameters()).device,
        )

    if last_saved_step != args.steps:
        _save_state(Path(args.save_dir), model, optimizer, args.steps, epoch, sampler_cursor, config)

    if eval_packs is not None:
        model.eval()
        with torch.no_grad():
            losses = []
            for samples in eval_packs:
                batch = _pack_replay_samples(samples)
                value = _loss_for_batch(
                    model, batch, samples, data_parallel_size, objective=args.loss_objective, dp_group=plugin.dp_group
                ).detach()
                dist.all_reduce(value, group=plugin.dp_group)
                losses.append(float((value / data_parallel_size).cpu()))
                clear_cosmos3_packed_meta(model, strict=True)
        if rank == 0:
            print(json.dumps({"heldout_losses": losses}), flush=True)

    if metrics is not None:
        timed = step_times[args.bench_warmup :] or step_times
        ordered = sorted(timed)
        summary = {"steps": args.steps - start_step}
        if ordered:
            summary.update(
                {
                    "step_ms_median": ordered[len(ordered) // 2] * 1e3,
                    "step_ms_min": ordered[0] * 1e3,
                    "step_ms_mean": sum(timed) / len(timed) * 1e3,
                    "timing_sources": {"step_ms": next(iter(timing_sources)) if len(timing_sources) == 1 else "mixed"},
                }
            )
        if eval_packs is not None:
            summary["heldout_losses"] = losses
        metrics.finish(summary)

    dist.destroy_process_group()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
