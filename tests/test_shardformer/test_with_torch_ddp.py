from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


@parameterize("lazy_init", [True, False])
def check_shardformer_with_ddp(lazy_init: bool):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_gpt", exclude="transformers_gptj")

    # create shardformer
    # ranks: [0, 1, 2, 3]
    # tp ranks = [0, 1], [2, 3]
    # dp ranks = [0, 2], [1, 3]
    dp_process_group_1 = dist.new_group([0, 2])
    dp_process_group_2 = dist.new_group([1, 3])
    tp_process_group_1 = dist.new_group([0, 1])
    tp_process_group_2 = dist.new_group([2, 3])

    coordinator = DistCoordinator()

    if coordinator.rank in [0, 1]:
        tp_process_group = tp_process_group_1
    else:
        tp_process_group = tp_process_group_2

    if coordinator.rank in [0, 2]:
        dp_process_group = dp_process_group_1
    else:
        dp_process_group = dp_process_group_2

    shard_config = ShardConfig(tensor_parallel_process_group=tp_process_group, enable_fused_normalization=True)
    shardformer = ShardFormer(shard_config=shard_config)

    ctx = LazyInitContext() if lazy_init else nullcontext()

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        # create and shard model
        with ctx:
            model = model_fn().cuda()
        sharded_model, _ = shardformer.optimize(model)

        # add ddp
        sharded_ddp_model = DDP(sharded_model, process_group=dp_process_group)

        # prepare input
        data = data_gen_fn()
        data = {k: v.cuda() for k, v in data.items()}

        # switch to train mode
        sharded_ddp_model.train()

        # run forward
        output = sharded_ddp_model(**data)
        loss = loss_fn(output)

        # backward
        loss.backward()
        torch.cuda.empty_cache()


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    check_shardformer_with_ddp()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_gpt2():
    spawn(run_dist, 4)


if __name__ == "__main__":
    test_gpt2()
