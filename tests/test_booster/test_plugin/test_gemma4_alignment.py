"""Gemma 4 E2B plugin alignment, token-weighted DP, export and exact resume.

No fused operators are substituted. BF16 recipes are explicit: DDP uses autocast
with FP32 parameters; ZeRO/Gemini use BF16 working weights with FP32 masters;
FSDP casts its FP32 master parameters to BF16 for forward/backward.
"""

import copy
import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import Gemma4ForCausalLM
from transformers import __version__ as transformers_version

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin, TorchFSDPPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import spawn
from tests.kit.model_zoo.transformers.gemma4 import gemma4_data, tiny_gemma4
from tests.kit.numerics import assert_aligned, full_precision_matmul

RECIPES = ("ddp_fp32", "ddp_bf16", "zero1_bf16", "zero2_bf16", "gemini_bf16", "fsdp_fp32", "fsdp_bf16")


def _clone(value):
    return value.detach().cpu().clone()


def _check_mapping(values, expected):
    assert expected and set(values) == set(expected), "Missing or unexpected parameter trace keys"
    for name, value in values.items():
        assert tuple(value.shape) == expected[name], name
        assert torch.isfinite(value).all(), name


def _bf16_working(recipe):
    return recipe.endswith("bf16") and not recipe.startswith("ddp")


def _build(initial, recipe):
    native = copy.deepcopy(initial).cuda().train()
    if _bf16_working(recipe):
        # All full-BF16 recipes start with identical rounded working/master values.
        native.bfloat16().float()
    native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    native, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(native)
    assert native.get_input_embeddings().weight is native.get_output_embeddings().weight
    if recipe.startswith("ddp"):
        plugin = TorchDDPPlugin()
        booster = Booster(plugin=plugin, mixed_precision="bf16" if recipe.endswith("bf16") else None)
    elif recipe.startswith("zero"):
        plugin = LowLevelZeroPlugin(
            stage=int(recipe[4]), precision="bf16", cpu_offload=True, max_norm=0, overlap_communication=False
        )
        booster = Booster(plugin=plugin)
    elif recipe.startswith("gemini"):
        plugin = GeminiPlugin(
            precision="bf16",
            placement_policy="static",
            min_chunk_size_m=1,
            initial_scale=1,
            enable_all_optimization=False,
            enable_async_reduce=False,
            enable_gradient_accumulation=True,
        )
        booster = Booster(plugin=plugin)
    else:
        mixed = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        plugin = TorchFSDPPlugin(mixed_precision=mixed if recipe.endswith("bf16") else None)
        # Required by FSDP's public summon_full_params(with_grads=True) diagnostic API.
        plugin.fsdp_kwargs["use_orig_params"] = True
        booster = Booster(plugin=plugin)
    optimizer = (
        HybridAdam(native.parameters(), lr=1e-3, weight_decay=0.01)
        if recipe.startswith("gemini")
        else torch.optim.AdamW(native.parameters(), lr=1e-3, foreach=False)
    )
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in native.parameters()}
    model, optimizer, _, _, _ = booster.boost(model=native, optimizer=optimizer)
    if recipe.startswith("zero"):
        assert set(optimizer.working_to_master_param) == {id(p) for p in native.parameters()}
    return booster, model, optimizer, native


def _full_state(bundle, recipe):
    _, model, _, native = bundle
    if recipe.startswith("gemini"):
        values = model.state_dict(only_rank_0=False)
    elif recipe.startswith("fsdp"):
        with FSDP.summon_full_params(model.unwrap(), writeback=False):
            values = native.state_dict()
            values = {n: _clone(p) for n, p in values.items()}
    else:
        values = native.state_dict()
    return {n: _clone(p) for n, p in values.items()}


def _state(bundle, recipe, expected):
    values = _full_state(bundle, recipe)
    values = {n: _clone(values[n]) for n in expected}
    # Compare effective working weights across plugins; exact checkpoint tests below
    # additionally compare the saved optimizer state and the uninterrupted next update.
    if _bf16_working(recipe):
        values = {n: p.bfloat16() for n, p in values.items()}
    _check_mapping(values, expected)
    return values


def _gradients(bundle, recipe, expected):
    _, model, optimizer, native = bundle
    if recipe.startswith("zero"):
        values = {n: optimizer.get_param_grad(p) for n, p in native.named_parameters()}
        assert all(g is not None for g in values.values()), "Missing ZeRO gradient"
        values = {n: _clone(g) for n, g in values.items()}
    elif recipe.startswith("fsdp"):
        with FSDP.summon_full_params(model.unwrap(), writeback=False, with_grads=True):
            assert all(p.grad is not None for p in native.parameters()), "Missing FSDP gradient"
            values = {n: _clone(p.grad) for n, p in native.named_parameters()}
    elif recipe.startswith("gemini"):
        values, chunks = {}, {}
        manager = model.chunk_manager
        for name, param in native.named_parameters():
            assert getattr(param, "_gemini_reduced", False), f"Missing Gemini reduction: {name}"
            chunk = manager.get_chunk(param)
            grad_chunk = chunk if manager.reuse_fp16_chunk else chunk.grad_chunk
            assert grad_chunk is not None, name
            if grad_chunk not in chunks:
                payload = grad_chunk.payload.detach()
                if not grad_chunk.is_gathered:
                    shards = [torch.empty_like(payload) for _ in range(grad_chunk.pg_size)]
                    dist.all_gather(shards, payload, group=grad_chunk.torch_pg)
                    payload = torch.cat(shards)
                chunks[grad_chunk] = payload.cpu().clone()
            info = chunk.tensors_info[param]
            values[name] = chunks[grad_chunk][info.offset : info.end].reshape(expected[name]).clone()
    else:
        assert all(p.grad is not None for p in native.parameters()), "Missing DDP gradient"
        values = {n: _clone(p.grad) for n, p in native.named_parameters()}
    _check_mapping(values, expected)
    return values


def _masters(bundle, recipe, expected):
    _, _, optimizer, native = bundle
    if recipe.startswith("zero"):
        values = {}
        for name, param in native.named_parameters():
            shard = optimizer.working_to_master_param[id(param)].detach().cuda().flatten()
            group = optimizer.param_to_pg[param]
            shards = [torch.empty_like(shard) for _ in range(dist.get_world_size(group))]
            dist.all_gather(shards, shard, group=group)
            values[name] = _clone(torch.cat(shards)[: param.numel()].reshape(expected[name]))
    else:
        state = _full_state(bundle, recipe)
        values = {name: state[name] for name in expected}
    _check_mapping(values, expected)
    assert all(p.dtype == torch.float32 for p in values.values()), "Expected FP32 optimizer master values"
    return values


class _NativeReference:
    """Independent native HF forward plus explicit FP32-master AdamW recipe."""

    def __init__(self, initial, recipe, truth=False):
        self.recipe = recipe
        self.truth = truth
        self.model = copy.deepcopy(initial).cuda().train()
        if truth:
            self.model.to(torch.float64 if recipe.endswith("fp32") else torch.float32)
        elif _bf16_working(recipe):
            self.model.bfloat16()
        self.params = dict(self.model.named_parameters())
        self.masters = (
            {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in self.params.items()}
            if not truth and _bf16_working(recipe)
            else self.params
        )
        self.optimizer = torch.optim.AdamW(self.masters.values(), lr=1e-3, foreach=False)

    def context(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if not self.truth and self.recipe == "ddp_bf16"
            else nullcontext()
        )

    def backward(self, batch):
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        with self.context():
            output = self.model(**batch)
        result = {"logits": _clone(output.logits), "loss": _clone(output.loss)}
        if self.truth:
            output.loss.backward()
        else:
            # Single-process microbatch accumulation with actual causal target counts.
            total = batch["labels"][:, 1:].ne(-100).sum()
            for index in range(batch["input_ids"].shape[0]):
                micro = {k: v[index : index + 1] for k, v in batch.items()}
                weight = micro["labels"][:, 1:].ne(-100).sum() / total
                with self.context():
                    loss = self.model(**micro).loss * weight
                loss.backward()
        assert all(p.grad is not None for p in self.params.values())
        grads = {n: _clone(p.grad) for n, p in self.params.items()}
        return result, grads

    def step(self):
        if self.masters is not self.params:
            for name, param in self.params.items():
                self.masters[name].grad = param.grad.float().clone()
        self.optimizer.step()
        if self.masters is not self.params:
            with torch.no_grad():
                for name, param in self.params.items():
                    param.copy_(self.masters[name])
        return {n: _clone(p) for n, p in self.params.items()}


def _candidate_step(bundle, batch, recipe, expected):
    booster, model, optimizer, _ = bundle
    weight = batch.pop("loss_weight")
    optimizer.zero_grad()
    output = model(**batch)
    assert torch.isfinite(output.loss)
    global_loss = output.loss.detach().float() * weight
    dist.all_reduce(global_loss)
    booster.backward(output.loss * weight * dist.get_world_size(), optimizer)
    grads = _gradients(bundle, recipe, expected)
    optimizer.step()
    return {"logits": _clone(output.logits), "loss": _clone(global_loss)}, grads, _state(bundle, recipe, expected)


def _assert_equal_state(left, right):
    assert left and set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)


def _optimizer_state(bundle, recipe):
    _, model, optimizer, _ = bundle
    if recipe.startswith("gemini"):
        state = optimizer.state_dict(only_rank_0=False)
    elif recipe.startswith("fsdp"):
        state = FSDP.full_optim_state_dict(model.unwrap(), optimizer.optim, rank0_only=False)
    else:
        state = optimizer.state_dict()
    assert state["state"], "Empty AdamW optimizer state after three updates"
    flattened = {}

    def visit(value, path):
        if torch.is_tensor(value):
            flattened[path] = _clone(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}/{key}")
        elif isinstance(value, (int, float, bool)):
            flattened[path] = torch.tensor(value)

    visit(state["state"], "state")
    assert any(name.endswith("/exp_avg") for name in flattened)
    assert any(name.endswith("/exp_avg_sq") for name in flattened)
    assert any(name.endswith("/step") for name in flattened)
    return flattened


@full_precision_matmul()
def _run_alignment(rank, world_size, port, recipe, directory):
    torch.set_num_threads(1)
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="127.0.0.1", seed=42)
    try:
        torch.manual_seed(42)
        initial = tiny_gemma4("e2b")
        expected = {n: tuple(p.shape) for n, p in initial.named_parameters()}
        assert any("embed_tokens_per_layer" in n for n in expected)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "scenario": "gemma4_e2b_l2",
                        "recipe": recipe,
                        "world_size": world_size,
                        "torch": torch.__version__,
                        "transformers": transformers_version,
                        "cuda_runtime": torch.version.cuda,
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                        "expected_gradient_tensors": len(expected),
                        "planned_training_steps": 3,
                    }
                ),
                flush=True,
            )
        truth = _NativeReference(initial, recipe, truth=True)
        reference = _NativeReference(initial, recipe)
        bundle = _build(initial, recipe)
        full = {k: v.cuda() for k, v in gemma4_data().items()}
        total_tokens = full["labels"][:, 1:].ne(-100).sum()
        local = {k: v[rank::world_size] for k, v in full.items()}
        local["loss_weight"] = local["labels"][:, 1:].ne(-100).sum() / total_tokens
        curves = [[], [], []]
        for step in range(3):
            truth_out, truth_grad = truth.backward(full)
            ref_out, ref_grad = reference.backward(full)
            # Metadata is not a model forward input.
            weight = local.pop("loss_weight")
            booster, model, optimizer, _ = bundle
            optimizer.zero_grad()
            output = model(**local)
            loss = output.loss.detach().float() * weight
            dist.all_reduce(loss)
            booster.backward(output.loss * weight * world_size, optimizer)
            candidate_grad = _gradients(bundle, recipe, expected)
            _check_mapping(truth_grad, expected)
            _check_mapping(ref_grad, expected)
            assert_aligned(candidate_grad, ref_grad, truth_grad)
            candidate_out = {"logits": _clone(output.logits), "loss": _clone(loss)}
            for result in (ref_out, truth_out):
                result["logits"] = result["logits"][rank::world_size]
            assert_aligned(candidate_out, ref_out, truth_out)
            optimizer.step()
            values = _state(bundle, recipe, expected)
            truth_values, ref_values = truth.step(), reference.step()
            assert_aligned(values, ref_values, truth_values)
            masters = _masters(bundle, recipe, expected)
            reference_masters = {n: _clone(p) for n, p in reference.masters.items()}
            assert_aligned(masters, reference_masters, truth_values)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "recipe": recipe,
                            "world_size": world_size,
                            "step": step + 1,
                            "loss": candidate_out["loss"].item(),
                            "reference_loss": ref_out["loss"].item(),
                            "truth_loss": truth_out["loss"].item(),
                            "aligned_gradient_tensors": len(candidate_grad),
                            "aligned_master_tensors": len(masters),
                        }
                    ),
                    flush=True,
                )
            for curve, result in zip(curves, (candidate_out, ref_out, truth_out)):
                curve.append(result["loss"])
            local["loss_weight"] = weight
        assert_aligned(*(torch.stack(curve) for curve in curves))
        # Save through each plugin's checkpoint I/O, then compare the next update
        # against the uninterrupted original, not a rounded HF-only checkpoint.
        checkpoint = Path(directory)
        saved_optimizer_state = _optimizer_state(bundle, recipe)
        booster.save_model(model, str(checkpoint / "model.pt"))
        booster.save_optimizer(optimizer, str(checkpoint / "optimizer.pt"))
        dist.barrier()
        continued = _candidate_step(bundle, dict(local), recipe, expected)
        restored = _build(initial, recipe)
        restored[0].load_model(restored[1], str(checkpoint / "model.pt"))
        restored[0].load_optimizer(restored[2], str(checkpoint / "optimizer.pt"))
        _assert_equal_state(_optimizer_state(restored, recipe), saved_optimizer_state)
        resumed = _candidate_step(restored, dict(local), recipe, expected)
        for actual, wanted in zip(resumed, continued):
            _assert_equal_state(actual, wanted)
        # Explicit HF export of gathered plugin parameters with alias restoration.
        state = _state(restored, recipe, expected)
        exported_state = _full_state(restored, recipe)
        if rank == 0:
            export = copy.deepcopy(initial).to(next(iter(state.values())).dtype)
            export.load_state_dict(exported_state, strict=True)
            assert export.get_input_embeddings().weight is export.get_output_embeddings().weight
            export.save_pretrained(checkpoint / "hf")
            loaded = Gemma4ForCausalLM.from_pretrained(
                checkpoint / "hf", attn_implementation="eager", dtype=next(export.parameters()).dtype
            )
            # HF initializes non-persistent rotary buffers in FP32 even when
            # loading BF16 parameters. Reapply the source's whole-model dtype
            # recipe, including those reconstructed buffers, before comparison.
            loaded.to(next(export.parameters()).dtype)
            export.eval()
            loaded.eval()
            assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight
            _assert_equal_state(dict(loaded.named_parameters()), dict(export.named_parameters()))
            _assert_equal_state(dict(loaded.named_buffers()), dict(export.named_buffers()))
            with torch.no_grad():
                cpu_batch = gemma4_data()
                torch.testing.assert_close(loaded(**cpu_batch).logits, export(**cpu_batch).logits, rtol=0, atol=0)
            print(
                json.dumps(
                    {
                        "recipe": recipe,
                        "world_size": world_size,
                        "strict_optimizer_state_resume": "passed",
                        "uninterrupted_next_step": "bitwise_equal",
                        "hf_reload": "bitwise_equal",
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()
        get_accelerator().empty_cache()


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("world_size", [1, 2])
def test_gemma4_plugin_alignment(recipe, world_size, tmp_path):
    assert get_accelerator().device_count() >= world_size, f"Requires {world_size} allocated devices"
    spawn(_run_alignment, world_size, recipe=recipe, directory=str(tmp_path))
