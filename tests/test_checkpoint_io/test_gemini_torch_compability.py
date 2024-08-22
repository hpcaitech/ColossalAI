import pytest
import torch
import torch.distributed as dist
from torch.optim import Adam
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, TorchDDPPlugin
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
@parameterize("shard", [False, True])
@parameterize("model_name", ["transformers_llama_for_causal_lm"])
def exam_torch_load_from_gemini(shard: bool, model_name: str):
    (model_fn, data_gen_fn, output_transform_fn, _, _) = next(iter(model_zoo.get_sub_registry(model_name).values()))
    criterion = lambda x: x.mean()
    plugin = GeminiPlugin(precision="fp16", initial_scale=(2**14))
    booster = Booster(plugin=plugin)

    model = model_fn()
    optimizer = HybridAdam(model.parameters(), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    data = data_gen_fn()
    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}
    output = model(**data)
    output = output_transform_fn(output)
    output_key = list(output.keys())[0]
    loss = criterion(output[output_key])

    booster.backward(loss, optimizer)
    optimizer.step()

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"

        booster.save_model(model, model_ckpt_path, shard=shard)
        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=shard)
        dist.barrier()

        new_model = model_fn()
        new_optimizer = Adam(new_model.parameters(), lr=0.001)
        new_plugin = TorchDDPPlugin()
        new_booster = Booster(plugin=new_plugin)
        new_model, new_optimizer, criterion, _, _ = new_booster.boost(new_model, new_optimizer, criterion)

        # Loading HybridAdam states to torch.Adam
        new_booster.load_model(new_model, model_ckpt_path, strict=True)

        # Add prefix to get aligned with pytorch parameter names.
        check_state_dict_equal(
            model.state_dict(only_rank_0=False, prefix="module.module."),
            new_model.state_dict(),
            ignore_device=False,
            ignore_dtype=True,
        )

        new_booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        check_state_dict_equal(optimizer.state_dict(only_rank_0=False), new_optimizer.state_dict(), ignore_device=False)

        # Check the new model/optimizer can successfully run.
        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }
        output = new_model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])
        new_booster.backward(loss, new_optimizer)
        new_optimizer.step()
        new_booster.save_model(new_model, model_ckpt_path, shard=shard)
        new_booster.save_optimizer(new_optimizer, optimizer_ckpt_path, shard=shard)


@clear_cache_before_run()
@parameterize("shard", [False, True])
@parameterize("model_name", ["transformers_gpt"])
def exam_gemini_load_from_torch(shard: bool, model_name: str):
    (model_fn, data_gen_fn, output_transform_fn, _, _) = next(iter(model_zoo.get_sub_registry(model_name).values()))
    criterion = lambda x: x.mean()
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    model = model_fn()
    optimizer = Adam(model.parameters(), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    data = data_gen_fn()
    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}
    output = model(**data)
    output = output_transform_fn(output)
    output_key = list(output.keys())[0]
    loss = criterion(output[output_key])

    booster.backward(loss, optimizer)
    optimizer.step()

    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"

        booster.save_model(model, model_ckpt_path, shard=shard)
        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=shard)
        dist.barrier()

        new_model = model_fn()
        new_optimizer = HybridAdam(new_model.parameters(), lr=0.001)
        new_plugin = GeminiPlugin()
        new_booster = Booster(plugin=new_plugin)
        new_model, new_optimizer, criterion, _, _ = new_booster.boost(new_model, new_optimizer, criterion)

        # Loading torch.Adam states to HybridAdam
        new_booster.load_model(new_model, model_ckpt_path, strict=True)

        # Add prefix to get aligned with pytorch parameter names.
        check_state_dict_equal(
            new_model.state_dict(only_rank_0=False, prefix="module.module."),
            model.state_dict(),
            ignore_device=False,
            ignore_dtype=True,
        )

        new_booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        old_state_dict = optimizer.state_dict()
        new_state_dict = new_optimizer.state_dict(only_rank_0=False)

        # Comparison of param_groups needs special care here,
        # since not all hyperparameters in Adam are used by HybridAdam
        hyperparameters_to_examine = ["params", "lr", "betas", "eps", "weight_decay"]
        for old_group, new_group in zip(old_state_dict["param_groups"], new_state_dict["param_groups"]):
            for k in hyperparameters_to_examine:
                assert (
                    k in old_group and k in new_group
                ), f"Old group's keys: {list(old_group.keys())}, New group's keys: {list(new_group.keys())}"
                assert old_group[k] == new_group[k]
        check_state_dict_equal(old_state_dict["state"], new_state_dict["state"], ignore_device=False)

        # Check the new model/optimizer can successfully run.
        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }
        output = new_model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])
        new_booster.backward(loss, new_optimizer)
        new_optimizer.step()
        new_booster.save_model(new_model, model_ckpt_path, shard=shard)
        new_booster.save_optimizer(new_optimizer, optimizer_ckpt_path, shard=shard)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_torch_load_from_gemini()
    exam_gemini_load_from_torch()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_gemini_ckpIO(world_size):
    spawn(run_dist, world_size)
