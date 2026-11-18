"""Original-size official-weight Ouro forward agreement and one BF16 ZeRO step.

This is a real-weight smoke test, not a convergence or full-size L2 certificate.
"""

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.ouro import get_ouro_model_class
from colossalai.testing import spawn
from tests.kit.ouro import check_calls, count_call, reset_calls
from tests.kit.remote_models import REMOTE_MODELS, load_remote_classes


def manual_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
    """Run the pinned official loop with an independent triangular attention mask."""
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch, length = inputs_embeds.shape[:2]
    allowed = torch.ones(length, length, device=inputs_embeds.device, dtype=torch.bool).tril()
    allowed = allowed[None, None].expand(batch, 1, length, length)
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :].bool()
    mask = torch.zeros_like(allowed, dtype=inputs_embeds.dtype).masked_fill(
        ~allowed, torch.finfo(inputs_embeds.dtype).min
    )
    return self._official_forward(self, inputs_embeds=inputs_embeds, attention_mask={"full_attention": mask}, **kwargs)


def worker(rank, world_size, port, key, directory):
    torch.set_num_threads(2)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="127.0.0.1", seed=42)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        repo, revision = REMOTE_MODELS[key]
        path = snapshot_download(repo, revision=revision, local_files_only=True)
        _, Config, Original = load_remote_classes(key)
        config = Config.from_pretrained(path, local_files_only=True)
        config.use_cache = False
        config._attn_implementation = "eager"
        Model = get_ouro_model_class(Original)
        model, info = Model.from_pretrained(
            path,
            config=config,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"], info
        parameter_count = sum(p.numel() for p in model.parameters())
        assert parameter_count > 1_000_000_000
        print("FULL_MODEL_LOADED", key, parameter_count, flush=True)
        model = model.cuda().eval()
        ids = torch.randint(5, 128, (1, 32), device="cuda")
        batch = dict(input_ids=ids, labels=ids.clone(), use_cache=False)
        reference = copy.deepcopy(model)
        reference.model._official_forward = sys.modules[Original.__module__].OuroModel.forward
        reference.model.forward = MethodType(manual_forward, reference.model)
        with torch.no_grad():
            expected = reference(**batch)
            actual = model(**batch)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
            torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        del reference, expected, actual
        torch.cuda.empty_cache()
        print("FULL_FORWARD_BITWISE_EQUAL", key, flush=True)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in model.model.layers:
            layer.register_forward_pre_hook(count_call)
        model.register_forward_pre_hook(reset_calls)
        model.register_forward_hook(check_calls)
        model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
        native = model
        before = {n: p.detach().flatten()[:32].cpu().clone() for n, p in native.named_parameters()}
        booster = Booster(plugin=LowLevelZeroPlugin(stage=2, precision="bf16"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        model, optimizer, *_ = booster.boost(model, optimizer)
        output = model(**batch)
        assert torch.isfinite(output.loss)
        booster.backward(output.loss, optimizer)
        count = 0
        for name, param in native.named_parameters():
            grad = optimizer.get_param_grad(param)
            assert grad is not None and torch.isfinite(grad).all(), name
            count += 1
        assert all(layer._ouro_calls > config.total_ut_steps for layer in native.model.layers)
        optimizer.step()
        changed = [
            n for n, p in native.named_parameters() if not torch.equal(before[n], p.detach().flatten()[:32].cpu())
        ]
        assert changed, "No parameter updates"
        result = dict(
            status="passed",
            model=repo,
            revision=revision,
            parameter_count=parameter_count,
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            loops=config.total_ut_steps,
            dtype="bfloat16",
            batch_size=1,
            sequence_length=32,
            reference="same official weights and loop, independently constructed causal mask",
            full_forward="bitwise_equal",
            optimizer="ZeRO-2 AdamW, one GPU, one step",
            gradient_tensors=count,
            changed_parameter_samples=len(changed),
            checkpoint_recompute=True,
            loss=float(output.loss.detach()),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.monotonic() - started,
            scope="original-size pretrained smoke; not full-size optimizer/gradient L2, long-context, convergence or performance acceptance",
        )
        (Path(directory) / f"{key}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ouro14", "ouro26"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    try:
        spawn(worker, 1, key=args.model, directory=args.out)
    except Exception:
        (Path(args.out) / f"{args.model}.json").write_text(
            json.dumps({"status": "failed", "traceback": traceback.format_exc()}, indent=2)
        )
        raise


if __name__ == "__main__":
    main()
