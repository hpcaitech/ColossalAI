import os
import shutil
from copy import deepcopy
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
import transformers
from packaging.version import Version

import colossalai
from colossalai.booster.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from tests.test_moe.moe_utils import assert_loose_close, check_model_equal

NUM_BATCH = 8
NUM_TOK_PER_BATCH, NUM_EXPERTS = 64, 4
NUM_LAYERS = 4
HIDDEN_SIZE_PER_HEAD = 4
NUM_HEADS = 8
NUM_KV_HEADS = 4
TOP_K = 2
VOCAB_SIZE = 128


def make_config():
    from transformers import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS,
        head_dim=HIDDEN_SIZE_PER_HEAD,
        intermediate_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS * 2,
        moe_intermediate_size=HIDDEN_SIZE_PER_HEAD * NUM_HEADS,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
        # layer 1 keeps a dense MLP, the others are sparse
        mlp_only_layers=[1],
        vocab_size=VOCAB_SIZE,
        attn_implementation="sdpa",
    )


def make_plugin(stage, ep_size, pp_size, tp_size, sp_size):
    return MoeHybridParallelPlugin(
        pp_size=pp_size,
        num_microbatches=pp_size,
        tp_size=tp_size,
        sp_size=sp_size,
        ep_size=ep_size,
        zero_stage=stage,
        enable_sequence_parallelism=sp_size > 1,
        sequence_parallelism_mode="all_to_all" if sp_size > 1 else None,
        overlap_communication=False,
        initial_scale=1,
        precision="bf16",
        find_unused_parameters=True,
    )


def run_qwen3_moe_commom(config: Tuple[int, ...]):
    from transformers import Qwen3MoeModel

    Randomizer.reset_index()
    stage, ep_size, pp_size, tp_size, sp_size = config
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    dtype = torch.bfloat16
    torch.cuda.set_device(dist.get_rank())

    plugin = make_plugin(stage, ep_size, pp_size, tp_size, sp_size)
    dp_size = plugin.dp_size
    booster = Booster(plugin=plugin)

    assert pp_size <= NUM_LAYERS, "pp_size should be less than or equal to NUM_LAYERS"
    model_config = make_config()

    # init model with the same seed
    seed_all(10086)

    torch_model = Qwen3MoeModel(model_config).to(dtype).cuda()
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1)

    parallel_model = deepcopy(torch_model)
    parallel_optimizer = torch.optim.SGD(parallel_model.parameters(), lr=1)
    parallel_model, parallel_optimizer, _, _, _ = booster.boost(parallel_model, parallel_optimizer)

    # create different input along dp axis
    seed_all(1453 + rank)

    torch_model.train()
    parallel_model.train()
    for _ in range(2):
        # gen random input
        input_embeddings = torch.rand(
            NUM_BATCH, NUM_TOK_PER_BATCH, HIDDEN_SIZE_PER_HEAD * NUM_HEADS, requires_grad=True
        ).cuda()
        dist.all_reduce(
            input_embeddings, group=plugin.pp_group
        )  # pp inputs except the first stage doesn't matter, but need to be replicate for torch model check

        dist.all_reduce(input_embeddings, group=plugin.tp_group)  # tp group duplicate input
        dist.all_reduce(input_embeddings, group=plugin.sp_group)  # sp group duplicate input

        # run the model with hybrid parallel
        if booster.plugin.stage_manager is not None:
            # for test with pp
            data_iter = iter([{"inputs_embeds": input_embeddings}])
            sharded_output = booster.execute_pipeline(
                data_iter,
                parallel_model,
                lambda x, y: x.last_hidden_state.mean(),
                parallel_optimizer,
                return_loss=True,
                return_outputs=True,
            )
            if booster.plugin.stage_manager.is_last_stage():
                parallel_output = sharded_output["loss"]
            else:
                parallel_output = torch.tensor(12345.0, device="cuda")

            # broadcast along pp axis
            dist.broadcast(
                parallel_output, src=dist.get_process_group_ranks(plugin.pp_group)[-1], group=plugin.pp_group
            )
        else:
            # for test without pp
            parallel_output = parallel_model(inputs_embeds=input_embeddings.to(dtype)).last_hidden_state.mean()
            parallel_optimizer.backward(parallel_output)
        parallel_optimizer.step()
        parallel_optimizer.zero_grad()
        dist.all_reduce(parallel_output, group=plugin.mixed_dp_group)

        # ===================================================================================
        # run normal model with all dp(different) inputs
        all_inputs = [torch.empty_like(input_embeddings) for _ in range(dp_size)]
        dist.all_gather(all_inputs, input_embeddings, group=plugin.mixed_dp_group)
        torch_output_sum = 0
        for input_data_ in all_inputs:
            torch_output = torch_model(inputs_embeds=input_data_.to(dtype)).last_hidden_state.mean()
            torch_output.backward()
            torch_output_sum += torch_output.detach()
        # avg dp grads follows zero optimizer
        for p in torch_model.parameters():
            if p.grad is not None:
                p.grad /= dp_size
        torch_optimizer.step()
        torch_optimizer.zero_grad()

        assert_loose_close(parallel_output, torch_output_sum, dtype=dtype)

    # use checkpoint to load sharded zero model
    model_dir = "./test_qwen3_moe"
    if rank == world_size - 1:
        os.makedirs(model_dir, exist_ok=True)

    dist.barrier()
    booster.save_model(parallel_model, model_dir, shard=True)
    dist.barrier()

    saved_model = Qwen3MoeModel.from_pretrained(model_dir).cuda().to(dtype)
    check_model_equal(torch_model, saved_model, dtype=dtype)
    dist.barrier()

    if rank == world_size - 1:
        shutil.rmtree(model_dir)

    print(f"rank {dist.get_rank()} test passed")


def run_qwen3_moe_causal_lm_commom(config: Tuple[int, ...]):
    # checks the causal lm head, the loss and the router aux loss, which is accumulated across pipeline stages
    from transformers import Qwen3MoeForCausalLM

    Randomizer.reset_index()
    stage, ep_size, pp_size, tp_size, sp_size = config
    rank = dist.get_rank()
    dtype = torch.bfloat16
    torch.cuda.set_device(dist.get_rank())

    plugin = make_plugin(stage, ep_size, pp_size, tp_size, sp_size)
    dp_size = plugin.dp_size
    booster = Booster(plugin=plugin)

    model_config = make_config()
    model_config.output_router_logits = True
    model_config.router_aux_loss_coef = 0.1
    # the transformers forward also collects the router logits (None) of dense layers and then fails in
    # load_balancing_loss_func, so only use sparse layers when computing the router aux loss
    model_config.mlp_only_layers = []

    seed_all(10086)
    torch_model = Qwen3MoeForCausalLM(model_config).to(dtype).cuda()
    parallel_model = deepcopy(torch_model)
    parallel_optimizer = torch.optim.SGD(parallel_model.parameters(), lr=1)
    parallel_model, parallel_optimizer, _, _, _ = booster.boost(parallel_model, parallel_optimizer)

    seed_all(1453 + rank)
    input_ids = torch.randint(0, VOCAB_SIZE, (NUM_BATCH, NUM_TOK_PER_BATCH), device="cuda")
    # replicate the input across every group but dp
    for group in (plugin.pp_group, plugin.tp_group, plugin.sp_group):
        dist.broadcast(input_ids, src=dist.get_process_group_ranks(group)[0], group=group)
    data = {"input_ids": input_ids, "labels": input_ids.clone()}

    if booster.plugin.stage_manager is not None:
        sharded_output = booster.execute_pipeline(
            iter([data]), parallel_model, lambda x, y: x.loss, parallel_optimizer, return_loss=True
        )
        if booster.plugin.stage_manager.is_last_stage():
            parallel_loss = sharded_output["loss"]
        else:
            parallel_loss = torch.tensor(12345.0, device="cuda")
        dist.broadcast(parallel_loss, src=dist.get_process_group_ranks(plugin.pp_group)[-1], group=plugin.pp_group)
    else:
        parallel_loss = parallel_model(**data).loss
    dist.all_reduce(parallel_loss, group=plugin.mixed_dp_group)

    all_input_ids = [torch.empty_like(input_ids) for _ in range(dp_size)]
    dist.all_gather(all_input_ids, input_ids, group=plugin.mixed_dp_group)
    # the pipeline schedule averages the loss over micro batches, the router aux loss is not linear in the batch,
    # so compute the reference loss on the same micro batches
    num_microbatches = pp_size if pp_size > 1 else 1
    with torch.no_grad():
        torch_loss = 0
        for ids in all_input_ids:
            for mb in ids.chunk(num_microbatches):
                torch_loss += torch_model(input_ids=mb, labels=mb).loss / num_microbatches

    assert_loose_close(parallel_loss, torch_loss, dtype=dtype)
    print(f"rank {dist.get_rank()} causal lm test passed")


@parameterize(
    "config",
    [
        # DDP: ep == 1 since ep * moe_dp == dp == moe_dp; sp == 1 since sp * dp == moe_dp == dp
        (0, 1, 4, 1, 1),
        (0, 1, 1, 4, 1),
        (0, 1, 2, 2, 1),
        # zero 1
        (1, 4, 1, 1, 1),
        (1, 1, 4, 1, 1),
        (1, 1, 1, 4, 1),
        (1, 2, 1, 1, 2),
        # zero 2
        (2, 4, 1, 1, 1),
        (2, 1, 4, 1, 1),
        (2, 1, 1, 4, 1),
        (2, 2, 1, 1, 2),
    ],
)
def run_qwen3_moe_test(config: Tuple[int, ...]):
    run_qwen3_moe_commom(config)


@parameterize(
    "config",
    [
        (1, 4, 1, 1, 1),
        (1, 1, 4, 1, 1),
        (1, 1, 1, 4, 1),
        (1, 2, 2, 1, 1),
    ],
)
def run_qwen3_moe_causal_lm_test(config: Tuple[int, ...]):
    run_qwen3_moe_causal_lm_commom(config)


@parameterize(
    "config",
    [
        # DDP: ep == 1 since ep * moe_dp == dp == moe_dp; sp == 1 since sp * dp == moe_dp == dp
        (0, 1, 2, 4, 1),
        (0, 1, 4, 2, 1),
        (0, 1, 1, 4, 1),
        (0, 1, 4, 1, 1),
        # zero 1:
        (1, 2, 1, 1, 2),
        (1, 2, 1, 4, 1),
        (1, 1, 1, 2, 2),
        (1, 2, 2, 2, 1),
        # zero 2
        (2, 2, 1, 1, 2),
        (2, 2, 1, 4, 1),
        (2, 1, 1, 2, 2),
        (2, 2, 2, 2, 1),
    ],
)
def run_qwen3_moe_3d_test(config: Tuple[int, ...]):
    print(f"{config=}")
    run_qwen3_moe_commom(config)


def check_qwen3_moe(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_qwen3_moe_test()
    run_qwen3_moe_causal_lm_test()


def check_qwen3_moe_3d(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_qwen3_moe_3d_test()


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.51.0"), reason="Requires transformers version 4.51.0 or later"
)
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_qwen3_moe(world_size):
    spawn(check_qwen3_moe, world_size)


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.51.0"), reason="Requires transformers version 4.51.0 or later"
)
@pytest.mark.largedist
@pytest.mark.parametrize("world_size", [8])
@rerun_if_address_is_in_use()
def test_qwen3_moe_3d(world_size):
    spawn(check_qwen3_moe_3d, world_size)


if __name__ == "__main__":
    test_qwen3_moe(world_size=4)
