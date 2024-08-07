import pytest
import torch
import torch.distributed as dist
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo


@clear_cache_before_run()
@parameterize("model_name", ["transformers_llama_for_causal_lm"])
@parameterize("plugin_type", ["ddp", "zero", "gemini"])
def exam_from_pretrained(plugin_type: str, model_name: str, shard=True, size_per_shard=32):
    (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) = next(
        iter(model_zoo.get_sub_registry(model_name).values())
    )
    criterion = loss_fn

    if plugin_type == "ddp":
        plugin = TorchDDPPlugin()
    elif plugin_type == "zero":
        plugin = LowLevelZeroPlugin(stage=2, max_norm=1.0, initial_scale=32)
    elif plugin_type == "gemini":
        plugin = GeminiPlugin(precision="fp16", initial_scale=32)
    else:
        raise ValueError(f"Plugin with type {plugin_type} is invalid, please check your argument.")

    booster = Booster(plugin=plugin)

    model = model_fn().cuda()
    model_huggingface_cls = model.__class__
    optimizer = HybridAdam(model.parameters(), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    data = data_gen_fn()
    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}
    output = model(**data)
    loss = criterion(output)

    booster.backward(loss, optimizer)
    optimizer.step()

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        booster.save_model(model, model_ckpt_path, shard=shard, size_per_shard=size_per_shard)
        dist.barrier()

        new_model = model_huggingface_cls.from_pretrained(model_ckpt_path)
        new_model = new_model.cuda()
        new_optimizer = HybridAdam(new_model.parameters(), lr=0.001)
        new_model, new_optimizer, criterion, _, _ = booster.boost(new_model, new_optimizer, criterion)

        if plugin_type == "gemini":
            check_state_dict_equal(model.state_dict(only_rank_0=False), new_model.state_dict(only_rank_0=False))
        else:
            check_state_dict_equal(model.unwrap().state_dict(), new_model.unwrap().state_dict())
        dist.barrier()


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_from_pretrained()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_huggingface_compatibility(world_size):
    spawn(run_dist, world_size)
