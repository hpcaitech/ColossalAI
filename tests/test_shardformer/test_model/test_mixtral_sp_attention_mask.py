from copy import deepcopy
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai._compat import is_transformers_v5
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all

# head_dim is set explicitly, so num_attention_heads * head_dim (4 * 8) differs from hidden_size.
HIDDEN_SIZE, NUM_HEADS, HEAD_DIM = 24, 4, 8
BATCH_SIZE, SEQ_LEN, NUM_PADDED = 2, 16, 5
# Relative L2 error allowed between the gathered parallel output and the unsharded one, both in bf16 (unit roundoff
# 3.9e-3). A wrong mask or reshape changes whole rows of the output, i.e. errors of order 1.
OUTPUT_RTOL = 2e-2


@parameterize(
    "config",
    [
        # (sp_mode, tp_size, sp_size) on 4 ranks; split_gather / ring are not supported yet
        ("all_to_all", 1, 2),
    ],
)
def run_sp_with_attention_mask(config: Tuple[str, int, int]):
    """Mixtral with sequence parallelism must accept a padding attention mask, like the unsharded model.

    The SP attention / model forwards replaced by the policy only run with sequence parallelism, and the existing
    Mixtral tests pass no attention mask.
    """
    sp_mode, tp_size, sp_size = config
    Randomizer.reset_index()
    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=tp_size,
        sp_size=sp_size,
        ep_size=1,
        # zero 0 would wrap the model in DDP, which the MoE plugin rejects when sp splits the dp group; ZeRO in turn
        # needs fp16 / bf16
        zero_stage=1,
        enable_sequence_parallelism=True,
        sequence_parallelism_mode=sp_mode,
        precision="bf16",
        overlap_communication=False,
        find_unused_parameters=True,
    )
    booster = Booster(plugin=plugin)

    seed_all(10086)
    model_config = MixtralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 2,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_local_experts=4,
        num_experts_per_tok=2,
        attn_implementation="eager",
    )
    torch_model = MixtralModel(model_config).to(torch.bfloat16).cuda()
    parallel_model = deepcopy(torch_model)
    optimizer = torch.optim.SGD(parallel_model.parameters(), lr=0.0)
    parallel_model, optimizer, _, _, _ = booster.boost(parallel_model, optimizer)

    # every rank uses the same batch, so the gathered parallel output equals the unsharded one
    seed_all(1453)
    inputs_embeds = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device="cuda")
    attention_mask[1, :NUM_PADDED] = 0  # left padding

    parallel_output = parallel_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
    optimizer.backward(parallel_output.mean())
    with torch.no_grad():
        torch_output = torch_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

    # sequence parallelism may leave each rank with its own part of the sequence
    if parallel_output.shape[1] != SEQ_LEN:
        parts = [torch.empty_like(parallel_output) for _ in range(dist.get_world_size(plugin.sp_group))]
        dist.all_gather(parts, parallel_output.detach().contiguous(), group=plugin.sp_group)
        parallel_output = torch.cat(parts, dim=1)
    error = ((parallel_output.detach().float() - torch_output.float()).norm() / torch_output.float().norm()).item()
    if dist.get_rank() == 0:
        print(f"{sp_mode} tp={tp_size} sp={sp_size}: relative output error {error:.3e}", flush=True)
    assert error <= OUTPUT_RTOL, f"{sp_mode} tp={tp_size} sp={sp_size}: relative output error {error:.3e}"


def run_unsupported_sp_mode():
    """split_gather / ring must be rejected when sharding, not hang in the MoE layer."""
    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=2,
        ep_size=1,
        zero_stage=1,
        enable_sequence_parallelism=True,
        sequence_parallelism_mode="split_gather",
        precision="bf16",
    )
    model_config = MixtralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 2,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    model = MixtralModel(model_config).to(torch.bfloat16).cuda()
    with pytest.raises(NotImplementedError, match="split_gather"):
        Booster(plugin=plugin).boost(model, torch.optim.SGD(model.parameters(), lr=0.0))


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    torch.cuda.set_device(rank)
    run_sp_with_attention_mask()
    if is_transformers_v5():
        run_unsupported_sp_mode()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_mixtral_sp_attention_mask():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_mixtral_sp_attention_mask()
