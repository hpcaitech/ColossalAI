from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all

# Relative error allowed between the ZeRO grad norm (bf16 grads) and the norm of an fp32 single-GPU reference. With
# the norm combined correctly it is below 1.2e-3 here; the old per-store norms were off by 0.6% to 67%.
NORM_RTOL = 5e-3

HIDDEN_SIZE = 8
NUM_EXPERTS = 8


@parameterize("tp_size", [1, 2])
@parameterize("ep_size", [2, 4])
@parameterize("moe_only", [True, False])
def run_grad_norm(tp_size: int, ep_size: int, moe_only: bool):
    """The ZeRO grad norm must be the norm of the whole gradient and equal on every rank.

    Dense and expert grads live in two ZeRO stores, and every EP rank holds different experts, so the norm has to
    combine both stores and all EP ranks. Training only the MoE blocks makes experts carry most of the norm, as in a
    real MoE model; training everything mixes dense and expert grads. Routing is random, so every expert is used on
    every moe_dp replica and the moe_dp scaling of #6464 cannot interfere. With tp, the norm must also count the
    tp-split params (fused MoE experts in transformers v5 are customized distributed tensors) on every tp rank.
    """
    if dist.get_world_size() % (ep_size * tp_size) != 0:
        return
    dtype = torch.bfloat16

    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=tp_size,
        ep_size=ep_size,
        zero_stage=1,
        precision="bf16",
        overlap_communication=False,
    )
    booster = Booster(plugin=plugin)

    seed_all(10086)
    config = MixtralConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE * 64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
    )
    torch_model = MixtralModel(config).to(dtype).cuda()
    for name, param in torch_model.named_parameters():
        # transformers v5 renamed `block_sparse_moe` to `mlp`
        param.requires_grad_(not moe_only or "block_sparse_moe" in name or ".mlp." in name)

    # single-GPU reference: same weights in fp32 on the concatenation of every data replica's batch (= global mean loss)
    single_model = deepcopy(torch_model).float()
    zero_model = deepcopy(torch_model)
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=0.0)
    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)

    # ranks of one tp group must see the same batch
    seed_all(1453 + dist.get_rank(plugin.dp_group))
    input_data = torch.rand(8, 16, HIDDEN_SIZE).cuda().to(dtype)
    zero_optimizer.backward(zero_model(inputs_embeds=input_data).last_hidden_state.mean())
    zero_optimizer.step()  # lr = 0: nothing moves, but this computes the norm used for clipping
    zero_norm = zero_optimizer.get_grad_norm()

    all_inputs = [torch.empty_like(input_data) for _ in range(dist.get_world_size(plugin.dp_group))]
    dist.all_gather(all_inputs, input_data, group=plugin.dp_group)
    single_model(inputs_embeds=torch.cat(all_inputs).float()).last_hidden_state.mean().backward()
    grads = [p.grad.double() for p in single_model.parameters() if p.grad is not None]
    ref_norm = sum(g.pow(2).sum() for g in grads).sqrt().item()

    norms = [None] * dist.get_world_size()
    dist.all_gather_object(norms, zero_norm)
    assert (
        len(set(norms)) == 1
    ), f"tp_size={tp_size} ep_size={ep_size} moe_only={moe_only}: grad norm differs across ranks: {norms}"
    assert abs(zero_norm - ref_norm) <= NORM_RTOL * ref_norm, (
        f"tp_size={tp_size} ep_size={ep_size} moe_only={moe_only}: ZeRO grad norm {zero_norm:.6g}, reference {ref_norm:.6g}, "
        f"ratio {zero_norm / ref_norm:.4f}"
    )


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    torch.cuda.set_device(rank)
    run_grad_norm()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_ep_zero_grad_norm(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_ep_zero_grad_norm(world_size=4)
