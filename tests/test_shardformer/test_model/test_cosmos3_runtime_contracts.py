"""CPU runtime regressions for precision, adaptive SAC and CP pack ownership."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from colossalai.shardformer.layer.two_way_attn import TwoWayPackedMeta
from colossalai.shardformer.modeling.cosmos3 import (
    Cosmos3ContextParallel,
    enable_cosmos3_selective_checkpointing,
    get_cosmos3_selective_checkpoint_stats,
)
from tests.test_shardformer.test_model.test_cosmos3_selective_checkpointing import _gradients, _Stack
from tests.test_shardformer.test_model.test_hsdp_and_context_parallel import (
    _check_context_parallel,
    _check_context_parallel_gradients,
)


def test_policy_postprocess_combines_precision_cp_and_selective():
    from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy
    from tests.test_shardformer.test_model.test_cosmos3_policy import Cosmos3PackedMoTAttention, _stand_in_shard_config

    model = _Stack()
    for layer in model.layers:
        layer.self_attn = Cosmos3PackedMoTAttention()
    model.time_embedder = torch.nn.Linear(8, 8)
    model.time_proj = torch.nn.Identity()
    model._cp_shard_fn = model._cp_gather_fn = None
    policy = Cosmos3OmniPolicy(selective_checkpointing=True)
    policy.set_model(model)
    policy.set_shard_config(
        _stand_in_shard_config(
            enable_sequence_parallelism=True,
            sequence_parallelism_mode="all_to_all",
            sequence_parallel_size=1,
            sequence_parallel_process_group=None,
        )
    )
    assert policy.postprocess() is model
    assert model._colossal_context_parallel is policy.context_parallel
    assert model._cp_shard_fn.__self__ is policy.context_parallel
    assert model.time_embedder._colossal_fp32_forward
    assert model._colossal_fp32_sinusoid_installed
    assert all(layer._colossal_selective_checkpointing for layer in model.layers)
    assert all(layer.self_attn.processor is policy.processor for layer in model.layers)


def test_adaptive_budget_reselects_for_changed_pack_and_preserves_gradients():
    torch.manual_seed(42)
    reference = _Stack()
    model = copy.deepcopy(reference)
    enable_cosmos3_selective_checkpointing(model, budget_gib=3000 / 2**30)
    for ar, dm, mode in [(2, 1, "mlp"), (20, 10, "sac"), (3, 2, "mlp")]:
        args = (torch.randn(ar, 8), torch.randn(dm, 8), (None,) * 4)
        expected = _gradients(reference, args)
        actual = _gradients(model, args)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want)
        stats = get_cosmos3_selective_checkpoint_stats(model)
        assert stats["mode"] == mode
        assert stats["estimated_layer_output_bytes"] == (ar + dm) * 8 * 4
        assert not stats["hard_budget"]


def test_full_selective_conflict_at_install_and_after_install():
    model = _Stack()
    model.gradient_checkpointing = True
    with pytest.raises(ValueError, match="cannot be combined"):
        enable_cosmos3_selective_checkpointing(model)
    model.gradient_checkpointing = False
    enable_cosmos3_selective_checkpointing(model)
    model.gradient_checkpointing = True
    with pytest.raises(ValueError, match="cannot be combined"):
        model(torch.randn(2, 8), torch.randn(3, 8), (None,) * 4)


def test_sac_observes_and_saves_real_cpu_flash_attention_ops():
    # CPU math SDPA decomposes into mm/softmax, so use the actual fused CPU dispatch to verify names and caching.
    from colossalai.shardformer.modeling.cosmos3 import _sac_checkpointed_decoder_forward, _SelectiveBudget

    class Layer(torch.nn.Module):
        def forward(self, und, gen, rotary):
            return tuple(
                torch.nn.functional.scaled_dot_product_attention(x[None, None], x[None, None], x[None, None])[0, 0]
                for x in (und, gen)
            )

    layer = Layer()
    layer._colossal_original_forward = layer.forward
    layer._colossal_layer_index = 0
    budget = _SelectiveBudget(1024 / 2**30, 2)
    und, gen = torch.randn(5, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    budget.decide(und, gen)
    assert budget.path == "sac" and budget.max_saved_layers > 0
    expected = layer(und, gen, None)
    actual = _sac_checkpointed_decoder_forward(layer, und, gen, (None,) * 4, budget)
    torch.testing.assert_close(actual, expected)
    torch.autograd.backward(actual, tuple(torch.ones_like(x) for x in actual))
    assert budget.stats["save_count"] == 2
    assert any("_scaled_dot_product_flash_attention_for_cpu" in name for name in budget.stats["attention_op_hits"])
    assert sum(budget.stats["attention_op_saves"].values()) == 2


def _cp_contract_worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        context = Cosmos3ContextParallel(process_group=dist.group.WORLD)
        # A group-local nonzero leader supplies nested, noncontiguous tensors, including fractional positions.
        inputs = (
            {
                "tokens": torch.arange(12).view(3, 4).t(),
                "positions": torch.tensor([[0.0, 0.25, 0.5]]),
                "payload": [torch.randn(5, 2), torch.empty(0)],
                "sample_ids": ["episode/α:3", "episode/b:7"],
            }
            if rank == 1
            else None
        )
        targets = (torch.arange(5.0), {"step": 7}) if rank == 1 else None
        meta = TwoWayPackedMeta([2, 1], [1, 4]) if rank == 1 else None
        inputs, targets, meta = context.broadcast_pack(inputs, targets, meta, leader=1, device="cpu")
        assert inputs["tokens"].shape == (4, 3)
        assert inputs["positions"].dtype == torch.float32
        assert inputs["positions"][0, 1] == 0.25
        assert targets[1]["step"] == 7
        assert inputs["sample_ids"] == ["episode/α:3", "episode/b:7"]
        assert context.packed_meta.ar_lens == (2, 1)
        context.validate_pack(
            tokens=inputs["tokens"],
            positions=inputs["positions"],
            payload=(inputs, targets),
        )
        # Each differing category must fail on all ranks, then the group remains usable.
        for category in ("tokens", "positions", "payload"):
            data = {
                "tokens": torch.zeros(3),
                "positions": torch.zeros(3),
                "payload": torch.zeros(3),
            }
            data[category][0] = rank
            with pytest.raises(ValueError, match="debug hash"):
                context.validate_pack(**data)
        with pytest.raises(ValueError, match="CP leader"):
            context.broadcast_pack(meta=None, leader=1)
        with pytest.raises(ValueError, match="CP leader"):
            context.broadcast_pack(inputs={"unpicklable": lambda: None} if rank == 1 else None, meta=meta, leader=1)
        with pytest.raises(ValueError, match="contract mismatch"):
            context.shard(
                torch.zeros(3, 8),
                torch.zeros(5, 8),
                (None,) * 4 if rank == 1 else tuple(torch.zeros(n, 4) for n in (3, 3, 5, 5)),
            )
        # Identical shapes with different sample boundaries are also rejected collectively before attention.
        context.set_packed_meta(TwoWayPackedMeta([2, 1] if rank == 0 else [1, 2], [1, 4]))
        with pytest.raises(ValueError, match="contract mismatch"):
            context.shard(
                torch.zeros(3, 8),
                torch.zeros(5, 8),
                tuple(torch.zeros(n, 4) for n in (3, 3, 5, 5)),
            )
        assert _check_context_parallel() < 1e-5
        assert _check_context_parallel_gradients() < 1e-5
    finally:
        dist.destroy_process_group()


def test_diffusers_backend_setting_is_effective_or_rejected(monkeypatch):
    from colossalai.shardformer.modeling.cosmos3 import ColossalTwoWayAttnProcessor

    monkeypatch.setattr(
        "colossalai.shardformer.modeling.cosmos3._load_cuda_varlen_attention",
        lambda: None,
    )
    processor = ColossalTwoWayAttnProcessor()
    processor._attention_backend = "cuda"
    assert processor.backend == "cuda"
    processor._attention_backend = "npu"
    assert processor.backend == "npu"
    with pytest.raises(ValueError, match="does not support diffusers backend"):
        processor._attention_backend = "flash_attention_2"
    assert processor.backend == "npu"


def test_precision_wrappers_compile_fullgraph_and_backward():
    from tests.test_shardformer.test_model.test_cosmos3_fp32_sinusoid import _installed, _timesteps

    torch.manual_seed(9)
    model = _installed()
    expected = model(_timesteps())
    expected.float().square().sum().backward()
    reference = [param.grad.clone() for param in model.parameters()]
    model.zero_grad(set_to_none=True)
    # AOT eager exercises forward/backward graph capture without requiring a local C++ compiler.
    # Capture and backward both run without outer autocast. PyTorch AOT's default backward_pass_autocast
    # assumes the capture autocast context also applies to backward; changing that context can round gradients
    # even though this wrapper disables autocast internally. Eager outer-autocast forward is checked separately.
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(model(_timesteps()), expected, rtol=0, atol=0)
    actual = compiled(_timesteps())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    for param, grad in zip(model.parameters(), reference):
        torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)
    # Repeated calls must consume a fresh sinusoidal projection instead of reusing a stale stash.
    torch.testing.assert_close(compiled(_timesteps() + 0.3), model(_timesteps() + 0.3), rtol=0, atol=0)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires gloo")
def test_cp_broadcast_debug_and_unequal_multisample_forward_backward(tmp_path):
    mp.spawn(_cp_contract_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
