import pytest
import torch
import torch.distributed as dist
from torch.optim import Adam
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import (
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo

TEST_CONFIGS = [
    (
        {"tp_size": 1, "pp_size": 2, "num_microbatches": 4, "zero_stage": 1, "precision": "fp16", "initial_scale": 1},
        {"tp_size": 2, "pp_size": 1, "num_microbatches": 4, "zero_stage": 1, "precision": "fp16", "initial_scale": 1},
    )
]


@parameterize("shard", [False, True])
@parameterize("model_name", ["transformers_llama_for_causal_lm"])
@parameterize("size_per_shard", [1])
@parameterize("test_config", TEST_CONFIGS)
@parameterize("use_async", [False, True])
@parameterize("low_cpu_mem_mode", [False, True])
@clear_cache_before_run()
def exam_state_dict(
    shard: bool, model_name: str, size_per_shard: int, test_config: dict, use_async: bool, low_cpu_mem_mode: bool
):
    (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) = next(
        iter(model_zoo.get_sub_registry(model_name).values())
    )
    criterion = loss_fn
    test_config_0, test_config_1 = test_config
    plugin_0 = HybridParallelPlugin(**test_config_0)
    booster_0 = Booster(plugin=plugin_0)

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    def _preprocess_data(data):
        if booster_0.plugin.stage_manager is not None:
            for k, v in data.items():
                if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                    new_shape = [1] * v.dim()
                    new_shape[0] = 4
                    data[k] = v.to("cuda").repeat(*new_shape)
            return iter([data])
        else:
            return {k: v.cuda() for k, v in data.items()}

    model_0 = model_fn().cuda()
    optimizer_0 = Adam(model_0.parameters(), lr=1e-3)
    model_0, optimizer_0, criterion, _, _ = booster_0.boost(model_0, optimizer_0, criterion)

    data = data_gen_fn()
    model_0.train()
    if booster_0.plugin.stage_manager is not None:
        booster_0.execute_pipeline(_preprocess_data(data), model_0, _criterion, optimizer_0, return_loss=True)
    else:
        output = model_0(**_preprocess_data(data))
        loss = criterion(output)
        optimizer_0.backward(loss)

    optimizer_0.step()
    optimizer_0.zero_grad()
    with shared_tempdir() as tempdir:
        model_ckpt_path_0 = f"{tempdir}/model_0"

        booster_0.save_model(
            model_0,
            model_ckpt_path_0,
            shard=shard,
            gather_dtensor=True,
            size_per_shard=size_per_shard,
            use_async=use_async,
        )
        booster_0.checkpoint_io._sync_d2h()
        booster_0.checkpoint_io._sync_io()
        dist.barrier()

        plugin_1 = HybridParallelPlugin(**test_config_1)
        booster_1 = Booster(plugin=plugin_1)

        model_1 = model_fn().cuda()
        optimizer_1 = Adam(model_1.parameters(), lr=1e-3)
        model_1, optimizer_1, criterion, _, _ = booster_1.boost(model_1, optimizer_1, criterion)

        booster_1.load_model(model_1, model_ckpt_path_0, low_cpu_mem_mode=low_cpu_mem_mode)

        model_ckpt_path_1 = f"{tempdir}/model_1"
        booster_1.save_model(
            model_1,
            model_ckpt_path_1,
            shard=shard,
            gather_dtensor=True,
            size_per_shard=size_per_shard,
            use_async=use_async,
        )
        booster_1.checkpoint_io._sync_d2h()
        booster_1.checkpoint_io._sync_io()
        dist.barrier()

        model_2 = model_fn().cuda()
        optimizer_2 = Adam(model_2.parameters(), lr=1e-3)
        model_2, optimizer_2, criterion, _, _ = booster_0.boost(model_2, optimizer_2, criterion)

        booster_0.load_model(model_2, model_ckpt_path_1, low_cpu_mem_mode=low_cpu_mem_mode)
        check_state_dict_equal(model_0.unwrap().state_dict(), model_2.unwrap().state_dict())

    dist.barrier()
    Randomizer.reset_index()
    clear_layout_converter()


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@rerun_if_address_is_in_use()
def test_hybrid_ckpIO(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_hybrid_ckpIO(4)
