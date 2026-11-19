"""CPU contracts for the Gemma 26B-A4B structural tiny model; not grouped CUDA acceptance."""

import copy

import pytest
import torch
import torch.distributed as dist

from colossalai.lazy import LazyInitContext
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.modeling.gemma4 import Gemma4GroupedExperts, replace_gemma4_experts
from colossalai.testing import rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.kit.model_zoo.transformers.gemma4 import gemma4_data, tiny_gemma4
from tests.kit.numerics.capture.training import prepare

ENTRY = "transformers_gemma4_26b_a4b_for_causal_lm"


def test_moe_structure_and_independent_reference():
    s = prepare(ENTRY, steps=2)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts, Gemma4TextRouter

    config = s.model.config
    assert config.enable_moe_block and config.num_experts == 8 and config.top_k_experts == 2
    assert config.hidden_size_per_layer_input == config.num_kv_shared_layers == 0
    assert config.attention_k_eq_v and config.num_global_key_value_heads < config.num_key_value_heads
    assert config.layer_types == ["sliding_attention"] * 5 + ["full_attention"]
    assert config.hidden_activation == "gelu_pytorch_tanh"
    assert s.model.is_gradient_checkpointing and s.reference.is_gradient_checkpointing
    for model, cls in (
        (s.model, Gemma4GroupedExperts),
        (s.reference, Gemma4TextExperts),
    ):
        assert model.model.embed_tokens.weight is model.lm_head.weight
        for layer in model.model.layers:
            assert isinstance(layer.experts, cls)
            assert isinstance(layer.router, Gemma4TextRouter)
            assert layer.experts.gate_up_proj.shape == (8, 64, 32)
            assert layer.experts.down_proj.shape == (8, 32, 32)
    for name, p in s.reference.state_dict().items():
        torch.testing.assert_close(p, s.model.state_dict()[name], rtol=0, atol=0)
    assert model_zoo[ENTRY][4].numerics.unused_params == ()


@pytest.mark.parametrize("lazy", [False, True])
def test_moe_policy_parameter_identity(lazy):
    if lazy:
        with LazyInitContext(default_device="cpu"):
            model = tiny_gemma4("26b_a4b")
    else:
        model = tiny_gemma4("26b_a4b")
    original = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    if lazy:
        LazyInitContext.materialize(model)
    assert original.keys() == dict(model.named_parameters()).keys()
    assert all(p is original[n] for n, p in model.named_parameters())
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model.parameters()}
    assert all(isinstance(l.experts, Gemma4GroupedExperts) for l in model.model.layers)
    assert all(l.experts.dispatcher.ep_size == 1 for l in model.model.layers)
    loss = model(**gemma4_data()).loss
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("checkpoint", [False, True])
def test_moe_cpu_multistep_and_hf_roundtrip(checkpoint, tmp_path):
    from transformers import Gemma4ForCausalLM

    torch.manual_seed(17)
    reference = tiny_gemma4("26b_a4b").double()
    candidate = replace_gemma4_experts(copy.deepcopy(reference))
    if checkpoint:
        candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (reference, candidate)]
    before = {n: p.detach().clone() for n, p in candidate.named_parameters()}
    for _ in range(3):
        for model, opt in zip((reference, candidate), opts):
            opt.zero_grad()
            loss = model(**gemma4_data()).loss
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        # FP32 router softmax and shared FP32 combine remain intentional even in a double model.
        torch.testing.assert_close(candidate(**gemma4_data()).loss, reference(**gemma4_data()).loss)
    changed = [n for n, p in candidate.named_parameters() if not torch.equal(p, before[n])]
    for pattern in (
        "router.proj",
        "router.scale",
        "router.per_expert_scale",
        "experts.gate_up_proj",
        "experts.down_proj",
        "mlp.",
    ):
        assert any(pattern in n for n in changed), pattern
    candidate.save_pretrained(tmp_path)
    restored = Gemma4ForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.float64,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    assert restored.model.embed_tokens.weight is restored.lm_head.weight
    for n, value in candidate.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[n], value, rtol=0, atol=0)
    torch.testing.assert_close(restored(**gemma4_data()).loss, candidate(**gemma4_data()).loss)


def _local_dispatch_in_dp(rank, world_size, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        from colossalai.moe import NCCLAllToAllDispatcher
        from colossalai.tensor.moe_tensor.api import is_moe_tensor

        model = replace_gemma4_experts(tiny_gemma4("26b_a4b"))
        assert all(l.experts.dispatcher.ep_size == 1 for l in model.model.layers)
        assert all(not is_moe_tensor(p) for p in model.parameters())
        # Different ranks need not have equal batches or participate in an EP collective.
        batch = {n: v[: rank + 1] for n, v in gemma4_data().items()}
        model(**batch).loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with pytest.raises(ValueError, match="local_only"):
            NCCLAllToAllDispatcher(8, dist.group.WORLD, local_only=True)
    finally:
        dist.destroy_process_group()


@rerun_if_address_is_in_use()
def test_moe_local_dispatch_inside_two_rank_dp():
    spawn(_local_dispatch_in_dp, 2)
