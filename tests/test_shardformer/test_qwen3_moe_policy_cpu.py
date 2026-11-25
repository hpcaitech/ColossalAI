"""Qwen3-MoE ShardFormer policy on CPU: dispatch, the native fallback without EP, and the EP path with gloo.

The tiny model has one sparse layer and one dense ``Qwen3MoeMLP`` layer (``mlp_only_layers=[1]``), because the policy has
to handle both kinds of ``mlp`` in the same model.
"""

import copy

import pytest
import torch
import torch.distributed as dist

from colossalai._compat import is_transformers_v5
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.testing import rerun_if_address_is_in_use, spawn

pytestmark = pytest.mark.skipif(not is_transformers_v5(), reason="Qwen3-MoE sharding needs transformers v5")

NUM_EXPERTS = 8
SPARSE = "model.layers.0.mlp.experts"


def _tiny_model(seed: int = 0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        mlp_only_layers=[1],
        max_position_embeddings=64,
        use_cache=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        experts_implementation="eager",
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config).float()
    for name, param in model.named_parameters():
        if name.endswith("mlp.gate.weight"):
            param.data.normal_()
        elif name.endswith("experts.gate_up_proj"):
            param.data.normal_(std=32**-0.5)
        elif name.endswith("experts.down_proj"):
            param.data.normal_(std=16**-0.5)
    return model


def test_policy_dispatch():
    from transformers import Qwen3MoeModel

    model = _tiny_model()
    assert type(get_autopolicy(model)).__name__ == "Qwen3MoeForCausalLMPolicy"
    assert type(get_autopolicy(Qwen3MoeModel(model.config))).__name__ == "Qwen3MoeModelPolicy"


def test_model_zoo_entries_run_forward_backward():
    from tests.kit.model_zoo import model_zoo

    entries = model_zoo.get_sub_registry("transformers_qwen3_moe")
    assert set(entries) == {"transformers_qwen3_moe", "transformers_qwen3_moe_for_causal_lm"}
    dense_entries = model_zoo.get_sub_registry("transformers_qwen3", exclude="transformers_qwen3_moe")
    assert all("qwen3_moe" not in name for name in dense_entries)

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in entries.items():
        model = model_fn()
        output = output_transform_fn(model(**data_gen_fn()))
        loss_fn(output).backward()
        layers = model.model.layers if hasattr(model, "model") else model.layers
        assert layers[0].mlp.experts.gate_up_proj.grad is not None, name


def test_without_ep_the_model_stays_native():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeTopKRouter

    model = _tiny_model()
    expected = copy.deepcopy(model)
    model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False)).optimize(model)
    assert isinstance(model.model.layers[0].mlp.experts, Qwen3MoeExperts)
    assert isinstance(model.model.layers[0].mlp.gate, Qwen3MoeTopKRouter)
    ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, expected(input_ids=ids).logits)


def _init_gloo(rank, world_size, port):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)


def _run_ep(rank, world_size, port):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

    from colossalai.moe import EPMoELayer
    from colossalai.tensor.moe_tensor.api import get_ep_group, is_moe_tensor, is_moe_tensor_ep_fused

    _init_gloo(rank, world_size, port)
    try:
        ep_group = dist.new_group(list(range(world_size)))
        model = _tiny_model()
        reference = copy.deepcopy(model)
        full_state = copy.deepcopy(model.state_dict())
        original = {name: param for name, param in model.named_parameters()}

        model, _ = ShardFormer(ShardConfig(enable_tensor_parallelism=False, ep_group=ep_group)).optimize(model)

        sparse = model.model.layers[0].mlp
        assert isinstance(sparse.experts, EPMoELayer)
        assert isinstance(sparse.gate, Qwen3MoeTopKRouter)  # the router keeps its class: the logits hook needs it
        assert not hasattr(model.model.layers[1].mlp, "experts")  # the dense layer is untouched
        local = NUM_EXPERTS // world_size

        # the Parameter objects are the original ones, now holding this rank's experts and tagged for EP and checkpoints
        for key in ("gate_up_proj", "down_proj"):
            param = getattr(sparse.experts, key)
            assert param is original[f"{SPARSE}.{key}"]
            assert param.shape[0] == local
            assert is_moe_tensor(param) and get_ep_group(param) is ep_group and is_moe_tensor_ep_fused(param)
            torch.testing.assert_close(
                param.data, full_state[f"{SPARSE}.{key}"][rank * local : (rank + 1) * local], rtol=0, atol=0
            )

        # forward on this rank's batch equals the native model on the same batch; the gradient reaches the original
        # Parameter objects (what optimizers and ZeRO hold)
        ids = torch.randint(1, 64, (world_size * 2, 8), generator=torch.Generator().manual_seed(1))
        mine = ids[rank * 2 : (rank + 1) * 2]
        out = model(input_ids=mine, labels=mine)
        out.loss.backward()
        with torch.no_grad():
            expected = reference(input_ids=mine).logits
        torch.testing.assert_close(out.logits.detach(), expected, rtol=1e-4, atol=1e-4)
        assert original[f"{SPARSE}.gate_up_proj"].grad is not None
        assert original[f"{SPARSE}.gate_up_proj"].grad.shape[0] == local

        # a checkpoint in the HF layout (all experts) loads into the sharded model through the layer's load hook
        for param in model.parameters():
            param.data.zero_()
        model.load_state_dict(full_state, strict=True)
        for key in ("gate_up_proj", "down_proj"):
            torch.testing.assert_close(
                getattr(sparse.experts, key).data,
                full_state[f"{SPARSE}.{key}"][rank * local : (rank + 1) * local],
                rtol=0,
                atol=0,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("ep_size", [2, 4])
@rerun_if_address_is_in_use()
def test_ep_replaces_only_the_experts(ep_size):
    spawn(_run_ep, ep_size)
