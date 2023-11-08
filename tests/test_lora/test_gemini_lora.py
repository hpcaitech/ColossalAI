import copy
import os

import torch
from peft import LoraConfig
from torch import distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    assert_equal,
    assert_not_equal,
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_checkpoint_io.utils import shared_tempdir

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0},  # zero2
    # {"placement_policy": "static", "shard_param_frac": 1.0},  # zero3
    # {"placement_policy": "static", "shard_param_frac": 0.5},  # zero3-half
    # {"placement_policy": "auto"},
]


@clear_cache_before_run()
def check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type, placement_config, master_weights):
    model = model_fn()
    model.gradient_checkpointing_enable()

    plugin = GeminiPlugin(
        max_norm=1.0,
        initial_scale=2**5,
        master_weights=master_weights,
        **placement_config,
    )
    booster = Booster(plugin=plugin)

    lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
    model = booster.enable_lora(model, lora_config=lora_config)
    model_copy = copy.deepcopy(model)

    optimizer = HybridAdam(model.parameters(), lr=0.001)
    criterion = loss_fn

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    for _ in range(2):
        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        output = model(**data)
        output = output_transform_fn(output)
        loss = criterion(output)

        booster.backward(loss, optimizer)
        optimizer.clip_grad_by_norm(1.0)
        optimizer.step()

    # Check parameters
    gemini_dict = model.state_dict(only_rank_0=False)
    model_copy_dict = model_copy.state_dict()

    for name, p1 in gemini_dict.items():
        p2 = model_copy_dict[name].to(p1.device).to(p1.dtype)
        if "lora_" in name:
            # lora modules require gradients, thus updated
            assert model.name2param[name].requires_grad
            assert_not_equal(p1, p2)
        else:
            modules_to_save = model.unwrap().modules_to_save
            if (modules_to_save is not None) and any(f"{key}.modules_to_save" in name for key in modules_to_save):
                # if a non-lora module should be saved, it should be updated
                assert model.name2param[name].requires_grad
                assert_not_equal(p1, p2)
            else:
                # if a non-lora module isn't supposed to be saved, it shouldn't be updated
                assert_equal(p1, p2)


@clear_cache_before_run()
def check_checkpoint(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type, placement_config, master_weights):
    plugin = GeminiPlugin(
        max_norm=1.0,
        initial_scale=2**5,
        master_weights=master_weights,
        **placement_config,
    )
    booster = Booster(plugin=plugin)
    lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
    criterion = loss_fn

    model_save = model_fn()
    model_load = copy.deepcopy(model_save)

    model_save = booster.enable_lora(model_save, lora_config=lora_config)
    optimizer_save = HybridAdam(model_save.parameters(), lr=0.001)
    model_save, optimizer_save, _, _, _ = booster.boost(model_save, optimizer_save)

    for _ in range(2):
        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        output = model_save(**data)
        output = output_transform_fn(output)
        loss = criterion(output)

        booster.backward(loss, optimizer_save)
        optimizer_save.clip_grad_by_norm(1.0)
        optimizer_save.step()

    with shared_tempdir() as tempdir:
        lora_ckpt_path = os.path.join(tempdir, "model_ckpt")
        optimizer_ckpt_path = os.path.join(tempdir, "optimizer_ckpt")

        booster.save_lora_as_pretrained(model_save, lora_ckpt_path)
        booster.save_optimizer(optimizer_save, optimizer_ckpt_path, shard=False)
        dist.barrier()

        # The Lora checkpoint should be small in size
        model_checkpoint_size_mb = os.path.getsize(os.path.join(lora_ckpt_path, "adapter_model.bin")) / (1024 * 1024)
        optimizer_checkpoint_size_mb = os.path.getsize(optimizer_ckpt_path) / (1024 * 1024)
        assert model_checkpoint_size_mb < 1 and optimizer_checkpoint_size_mb < 1

        new_plugin = GeminiPlugin(
            max_norm=1.0,
            initial_scale=2**5,
            master_weights=master_weights,
            **placement_config,
        )
        new_booster = Booster(plugin=new_plugin)
        model_load = new_booster.enable_lora(model_load, pretrained_dir=lora_ckpt_path)
        optimizer_load = HybridAdam(model_load.parameters(), lr=0.001)
        model_load, optimizer_load, _, _, _ = new_booster.boost(model_load, optimizer_load)

        booster.load_optimizer(optimizer_load, optimizer_ckpt_path)

        check_state_dict_equal(model_save.state_dict(only_rank_0=False), model_load.state_dict(only_rank_0=False))
        check_state_dict_equal(
            optimizer_save.state_dict(only_rank_0=False), optimizer_load.state_dict(only_rank_0=False)
        )


# TODO(Baizhou): add complete cases of parameters after tuning on basic case
@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("master_weights", [True])
def run_lora_test(placement_config: dict, master_weights: bool):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        task_type = None
        if name == "transformers_llama_for_casual_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"

        check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type, placement_config, master_weights)
        check_checkpoint(
            model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type, placement_config, master_weights
        )
        # check_grad_accum()
        break


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_lora_test()


@rerun_if_address_is_in_use()
def test_gemini_lora():
    spawn(run_dist, 2)
