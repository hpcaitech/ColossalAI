from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import assert_loose_close

NUM_BATCH = 4
NUM_TOK_PER_BATCH, NUM_EXPERTS = 7, 4
HIDDEN_SIZE_PER_HEAD = 4
NUM_HEADS = 4
TOP_K = 2


@parameterize("stage", [1])
@parameterize("ep_size", [2])
def run_zero_with_original_model(stage: int, ep_size: int):
    tp_size = dist.get_world_size() // ep_size
    dtype = torch.bfloat16

    rank = torch.distributed.get_rank()
    torch.cuda.set_device(dist.get_rank())

    seed_all(10086)

    config = MixtralConfig(
        hidden_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS,
        intermediate_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS * 2,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
    )
    torch_model = MixtralModel(config).to(dtype).cuda()

    zero_model = deepcopy(torch_model).to(dtype)
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)
    moe_booster = Booster(
        plugin=MoeHybridParallelPlugin(
            tp_size=tp_size,
            moe_tp_size=tp_size,
            pp_size=1,
            ep_size=ep_size,
            zero_stage=stage,
            overlap_communication=False,
            initial_scale=1,
        )
    )
    zero_model, zero_optimizer, _, _, _ = moe_booster.boost(zero_model, zero_optimizer)

    hybird_booster = Booster(
        plugin=HybridParallelPlugin(
            tp_size=tp_size,
            pp_size=1,
            zero_stage=stage,
            overlap_communication=False,
            initial_scale=1,
        )
    )
    hybrid_model, hybrid_optimizer, _, _, _ = hybird_booster.boost(
        torch_model, torch.optim.SGD(torch_model.parameters(), lr=1)
    )
    # create different input
    seed_all(1453 + rank)

    hybrid_model.train()
    zero_model.train()
    for _ in range(2):
        # zero-dp forward
        input_data = torch.rand(
            NUM_BATCH, NUM_TOK_PER_BATCH, HIDDEN_SIZE_PER_HEAD * NUM_HEADS, requires_grad=True
        ).cuda()
        zero_output = zero_model(inputs_embeds=input_data.to(dtype)).last_hidden_state.mean()
        # zero-dp backward
        zero_optimizer.backward(zero_output)
        # torch-ddp forward
        hybrid_output = hybrid_model(inputs_embeds=input_data.to(dtype)).last_hidden_state.mean()
        assert_loose_close(zero_output, hybrid_output, dtype=dtype)
        # torch-ddp backward
        hybrid_optimizer.backward(hybrid_output)

        # check grad
        name_to_p = {n: p for n, p in hybrid_model.named_parameters()}
        for n, p in zero_model.named_parameters():
            zero_grad = zero_optimizer.get_param_grad(p)
            if name_to_p[n].grad is None:
                name_to_p[n].grad = torch.zeros_like(name_to_p[n])
                continue
            if zero_grad.shape != name_to_p[n].grad.shape:  # TODO check sharded and sliced moe
                continue
            assert_loose_close(zero_grad, name_to_p[n].grad, dtype=dtype, name=n)

        # zero-dp step
        zero_optimizer.step()

        # original model step
        hybrid_optimizer.step()

        # check updated param
        for n, p in zero_model.named_parameters():
            if p.data.shape != name_to_p[n].data.shape:  # TODO check sharded and sliced moe
                continue
            assert_loose_close(p.data, name_to_p[n].data, dtype=dtype, name=n)

    print(f"{dist.get_rank()} test passed")


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_with_original_model()


@pytest.mark.skip("tested in corresponding sharderformer")
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_moe_ep_tp(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_ep_tp(world_size=4)
