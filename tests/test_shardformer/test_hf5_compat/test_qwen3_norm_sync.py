"""Qwen3 head-shared norm gradients, independently checked on CPU/Gloo."""

from types import SimpleNamespace

import pytest

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("HF5 Qwen3 norm contracts", allow_module_level=True)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Qwen3Config, Qwen3Model

from colossalai.shardformer.policies.qwen3 import Qwen3Policy


def _model():
    return Qwen3Model(
        Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )


def _policy(model, group, size):
    policy = Qwen3Policy()
    policy.model = model
    policy.shard_config = SimpleNamespace(
        enable_tensor_parallelism=size > 1,
        tensor_parallel_size=size,
        tensor_parallel_process_group=group,
    )
    return policy


def _worker(rank, rendezvous, dtype, seed):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(seed)
        reference = _model().to(dtype)
        actual = _model().to(dtype)
        actual.load_state_dict(reference.state_dict())
        policy = _policy(actual, dist.group.WORLD, 2)
        policy.postprocess()
        policy.postprocess()  # Repeated installation must not multiply gradients.
        ref_attention = reference.layers[0].self_attn
        attention = actual.layers[0].self_attn
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
        for step in range(2):
            if step == 1:
                actual.gradient_checkpointing_enable()
                policy.postprocess()
            ref_optimizer.zero_grad()
            optimizer.zero_grad()
            for _ in range(2):
                for name, heads in (("q_norm", 4), ("k_norm", 2)):
                    x = torch.randn(2, 3, heads, 8).to(dtype)
                    upstream = torch.randn_like(x) / 32
                    expected = getattr(ref_attention, name)(x)
                    local_input = x.chunk(2, 2)[rank].contiguous()
                    if step == 1:
                        from torch.utils.checkpoint import checkpoint

                        local = checkpoint(getattr(attention, name), local_input, use_reentrant=False)
                    else:
                        local = getattr(attention, name)(local_input)
                    expected.backward(upstream)
                    local.backward(upstream.chunk(2, 2)[rank].contiguous())
            for name in ("q_norm", "k_norm"):
                r, a = getattr(ref_attention, name).weight, getattr(attention, name).weight
                tolerance = 0.005 if dtype == torch.bfloat16 else 1e-5
                torch.testing.assert_close(a.grad, r.grad, atol=tolerance, rtol=tolerance)
                replicas = [torch.empty_like(a.grad) for _ in range(2)]
                dist.all_gather(replicas, a.grad)
                torch.testing.assert_close(replicas[0], replicas[1], atol=0, rtol=0)
            ref_optimizer.step()
            optimizer.step()
            for name in ("q_norm", "k_norm"):
                torch.testing.assert_close(
                    getattr(attention, name).weight, getattr(ref_attention, name).weight, atol=tolerance, rtol=tolerance
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_head_norm_gradient_sum(tmp_path, dtype, seed):
    mp.spawn(_worker, args=((tmp_path / "gloo-init").as_uri(), dtype, seed), nprocs=2, join=True)


def test_singleton_and_frozen_norms():
    model = _model()
    _policy(model, None, 1).postprocess()
    assert not getattr(model.layers[0].self_attn.q_norm.weight, "_qwen3_tp_grad_hook", False)
    for norm in (model.layers[0].self_attn.q_norm, model.layers[0].self_attn.k_norm):
        norm.weight.requires_grad_(False)
    _policy(model, None, 2).postprocess()
    assert not getattr(model.layers[0].self_attn.q_norm.weight, "_qwen3_tp_grad_hook", False)
