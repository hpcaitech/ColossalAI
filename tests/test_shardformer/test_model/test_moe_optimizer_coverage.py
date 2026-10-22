from typing import Tuple

import pytest
import torch
import torch.distributed as dist

from colossalai._compat import is_transformers_v5

if not is_transformers_v5():
    pytest.skip("fused MoE experts only exist in transformers v5", allow_module_level=True)

from transformers import DeepseekV3Config, MixtralConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all

NUM_TOKENS, HIDDEN = 64, 32


def _build(model_name: str):
    if model_name == "mixtral":
        config = MixtralConfig(
            hidden_size=HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            attn_implementation="eager",
        )
        return MixtralModel(config)
    if model_name == "mixtral_odd":
        # One expert per rank under ep_size 2 with odd sizes: the local fused experts have 1 x 34 x 33 = 1122 and
        # 1 x 33 x 17 = 561 elements, so padding them for moe_dp_size 2 (0 and 1) differs from padding them for the
        # global dp size 4 (2 and 3).
        config = MixtralConfig(
            hidden_size=33,
            intermediate_size=17,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_local_experts=2,
            num_experts_per_tok=2,
            attn_implementation="eager",
        )
        return MixtralModel(config)
    config = DeepseekV3Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        attn_implementation="eager",
    )
    return DeepseekV3Model(config)


def run_coverage(config: Tuple):
    """Every trainable parameter of the boosted model must be owned by the optimizer and updated.

    Numerical parity tests can miss parameters that silently fall out of the optimizer (e.g. MoE
    experts sharded into new Parameter objects) because small updates stay within loose tolerances.
    This check is structural. The update is checked on the fp32 master copy when the wrapper keeps
    one: a bf16 working weight near 1.0 (e.g. RMSNorm) can round a small update away.
    """
    model_name, zero_stage, ep_size, tp_size = config
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=tp_size,
        ep_size=ep_size,
        zero_stage=zero_stage,
        precision="bf16",
        initial_scale=1,
        overlap_communication=False,
        find_unused_parameters=True,
    )
    booster = Booster(plugin=plugin)
    seed_all(1234)
    model = _build(model_name).to(torch.bfloat16).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e3)
    model, optimizer, *_ = booster.boost(model, optimizer)

    # `embed_tokens` is unused because the test feeds `inputs_embeds`.
    params = [
        (name, p) for name, p in model.unwrap().named_parameters() if p.requires_grad and "embed_tokens" not in name
    ]
    label = f"{model_name} zero={zero_stage} ep={ep_size} tp={tp_size}"
    working_to_master = optimizer.get_working_to_master_map()
    if working_to_master is None:
        owned_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        tracked = {name: p for name, p in params if id(p) in owned_ids}
    else:
        tracked = {name: working_to_master[id(p)] for name, p in params if id(p) in working_to_master}
    not_owned = [name for name, _ in params if name not in tracked]
    assert not not_owned, f"{label}: not owned by the optimizer {not_owned}"

    before = {name: t.detach().float().clone() for name, t in tracked.items()}
    seed_all(4321 + rank)
    inputs = torch.rand(2, NUM_TOKENS // 2, HIDDEN, device="cuda", dtype=torch.bfloat16)
    loss = model(inputs_embeds=inputs).last_hidden_state.float().pow(2).mean()
    booster.backward(loss, optimizer)
    optimizer.step()

    unchanged = [name for name, t in tracked.items() if torch.equal(t.detach().float(), before[name])]
    assert not unchanged, f"{label}: not updated {unchanged}"
    assert any(".experts." in name for name, _ in params)


@parameterize(
    "config",
    [
        # (model, zero_stage, ep_size, tp_size) on 4 ranks
        ("mixtral", 1, 4, 1),
        ("mixtral", 1, 2, 2),
        ("mixtral", 0, 1, 2),
        ("deepseek_v3", 1, 4, 1),
    ],
)
def run_all(config: Tuple):
    run_coverage(config)


def check_coverage(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_all()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_moe_optimizer_coverage():
    spawn(check_coverage, 4)


if __name__ == "__main__":
    test_moe_optimizer_coverage()
