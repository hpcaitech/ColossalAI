import os
import shutil
from copy import deepcopy
from typing import Tuple

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
from tests.test_moe.moe_utils import loose_close
from tests.test_moe.test_moe_checkpoint import check_model_equal

NUM_BATCH = 4
NUM_TOK_PER_BATCH, NUM_EXPERTS = 7, 4
HIDDEN_SIZE_PER_HEAD = 4
NUM_HEADS = 4
TOP_K = 1


@parameterize("config", [(0, 1, 1), (0, 1, 2), (0, 1, 4), (1, 1, 4), (1, 2, 2), (1, 4, 1)])
def run_zero_with_original_model(config: Tuple[int, ...]):
    stage, ep_size, tp_size = config
    dtype, precision = torch.float16, "fp16"
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(dist.get_rank())

    plugin = MoeHybridParallelPlugin(
        pp_size=1,
        tp_size=tp_size,
        moe_tp_size=tp_size,
        ep_size=ep_size,
        zero_stage=stage,
        overlap_communication=False,
        initial_scale=1,
        precision=precision,
    )
    booster = Booster(plugin=plugin)

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
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)

    zero_model = deepcopy(torch_model).to(dtype)
    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)

    zero_model, zero_optimizer, _, _, _ = booster.boost(zero_model, zero_optimizer)

    # create different input
    seed_all(1453 + rank)

    torch_model.train()
    zero_model.train()
    for _ in range(2):
        input_data = torch.rand(
            NUM_BATCH, NUM_TOK_PER_BATCH, HIDDEN_SIZE_PER_HEAD * NUM_HEADS, requires_grad=True
        ).cuda()
        dist.all_reduce(input_data, group=plugin.tp_group)  # tp requires duplicate input

        zero_output = zero_model(inputs_embeds=input_data.to(dtype)).last_hidden_state.mean()
        zero_optimizer.backward(zero_output)
        zero_optimizer.step()
        zero_optimizer.zero_grad()
        dist.all_reduce(zero_output)

        all_inputs = [torch.empty_like(input_data) for _ in range(dist.get_world_size())]
        dist.all_gather(all_inputs, input_data)

        torch_output_sum = 0
        for input_data_ in all_inputs:
            torch_output = torch_model(inputs_embeds=input_data_.to(dtype)).last_hidden_state.mean()
            torch_output.backward()
            torch_output_sum += torch_output.detach()
        # avg dp grads
        for p in torch_model.parameters():
            if p.grad is not None:
                p.grad /= dist.get_world_size()
        torch_optimizer.step()
        torch_optimizer.zero_grad()

        loose_close(zero_output, torch_output_sum, dtype=dtype)

    # use checkpoint to load sharded zero model
    model_dir = "./test_mixtral"
    if dist.get_rank() == 0:
        os.makedirs(model_dir, exist_ok=True)

    dist.barrier()

    booster.save_model(zero_model, model_dir, shard=True)

    dist.barrier()

    saved_model = MixtralModel.from_pretrained(model_dir).cuda().to(dtype)
    check_model_equal(torch_model, saved_model)

    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(model_dir)

    print(f"{dist.get_rank()} test passed")


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_with_original_model()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_mistral(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_mistral(world_size=4)
