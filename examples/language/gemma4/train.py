"""Single-GPU, full-text-parameter Gemma 4 training with CPU optimizer offload."""

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.shardformer import ShardConfig, ShardFormer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local official snapshot directory (read only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--save", action="store_true", help="Save model and optimizer separately after training")
    args = parser.parse_args()
    if args.steps < 3 or args.sequence_length < 2:
        parser.error("Use at least three optimizer steps and two tokens")
    return args


def freeze_non_text(model):
    text_models = [m for m in model.modules() if isinstance(m, Gemma4TextModel)]
    if len(text_models) != 1:
        raise ValueError("Expected exactly one native Gemma 4 text backbone")
    model.requires_grad_(False)
    text_models[0].requires_grad_(True)
    model.lm_head.requires_grad_(True)
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("Expected tied token embedding and LM head")


def main():
    args = parse_args()
    result_dir = Path(args.output)
    result_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "args": vars(args),
        "steps": [],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "colossalai_file": colossalai.__file__,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "all text parameters; vision/audio frozen",
    }
    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise RuntimeError("Training evidence requires a clean, committed source checkout")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("This entry point is for single-GPU training only")
        accelerator = get_accelerator()
        if accelerator.device_count() != 1:
            raise ValueError("Expose exactly one allocated GPU to this process")
        colossalai.launch_from_torch(seed=args.seed)
        device = accelerator.get_current_device()
        accelerator.reset_peak_memory_stats()
        torch.manual_seed(args.seed)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=args.attention, local_files_only=True
        ).train()
        model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
        model.config.text_config.use_cache = False
        freeze_non_text(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        named = {name: p for name, p in model.named_parameters() if p.requires_grad}
        report["trainable_parameters"] = sum(p.numel() for p in named.values())
        report["frozen_parameters"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if not any("per_layer" in n and "embed" in n for n in named):
            raise AssertionError("PLE must remain trainable")
        report["frozen_names"] = [n for n, p in model.named_parameters() if not p.requires_grad]
        optim = torch.optim.AdamW(named.values(), lr=args.lr, foreach=False)
        assert {id(p) for g in optim.param_groups for p in g["params"]} == {id(p) for p in named.values()}
        booster = Booster(
            plugin=LowLevelZeroPlugin(
                stage=1,
                precision="bf16",
                cpu_offload=True,
                master_weights=True,
                max_norm=0,
                overlap_communication=False,
            )
        )
        model, optim, _, _, _ = booster.boost(model=model, optimizer=optim)
        assert set(optim.working_to_master_param) == {id(p) for p in named.values()}
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokens = tokenizer.encode(
            "The training example checks every text layer. A model learns by predicting the next token. "
        )
        ids = torch.tensor((tokens * (args.sequence_length // len(tokens) + 1))[: args.sequence_length], device=device)[
            None
        ]
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
        for step in range(args.steps):
            started = time.monotonic()
            optim.zero_grad()
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise AssertionError("Non-finite loss")
            booster.backward(loss, optim)
            witnesses = []
            missing = []
            for name, param in named.items():
                grad = optim.get_param_grad(param)
                if grad is None:
                    missing.append(name)
                    continue
                if not torch.isfinite(grad).all():
                    raise AssertionError(f"Non-finite gradient: {name}")
                # Inspect one touched embedding row, avoiding copies of the full PLE table.
                master = optim.working_to_master_param[id(param)]
                if master.device.type != "cpu" or master.dtype != torch.float32:
                    raise AssertionError(f"Expected CPU FP32 master: {name}")
                offset = int(ids[0, 0]) * param.shape[1] if param.ndim == 2 and "embed" in name else 0
                sample = master.flatten()[offset : offset + 32]
                witnesses.append((name, sample, sample.clone()))
            if missing:
                raise AssertionError(f"Missing text gradients: {missing}")
            optim.step()
            changed = [name for name, after, before in witnesses if not torch.equal(after, before)]
            if not changed or not any("per_layer" in n and "embed" in n for n in changed):
                raise AssertionError("No PLE master-weight update observed")
            accelerator.synchronize()
            record = {
                "step": step + 1,
                "loss": loss.item(),
                "seconds": time.monotonic() - started,
                "parameters_with_gradient": len(witnesses),
                "changed_samples": changed,
                "gpu_peak_bytes": accelerator.max_memory_allocated(),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
            report["steps"].append(record)
            print(json.dumps(record), flush=True)
        if args.save:
            booster.save_model(model, str(result_dir / "model"), shard=True, use_safetensors=True)
            booster.save_optimizer(optim, str(result_dir / "optimizer"), shard=True)
        report["status"] = "passed_training_steps"
        report["checkpoint_resume"] = "not tested"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        (result_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
