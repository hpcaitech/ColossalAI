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
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo
from tests.test_checkpoint_io.utils import shared_tempdir
from tests.test_lora.utils import check_param_equality, do_fwd_bwd

PLACEMENT_CONFIGS = [
    {"placement_policy": "static", "shard_param_frac": 0.0},  # zero2
    {"placement_policy": "static", "shard_param_frac": 1.0},  # zero3
    {"placement_policy": "static", "shard_param_frac": 0.5},  # zero3-half
    {"placement_policy": "auto"},
]


@clear_cache_before_run()
def check_fn(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type, placement_config, master_weights):
    model = model_fn()
    model.gradient_checkpointing_enable()
    model_load = copy.deepcopy(model)
    enable_gradient_accumulation = master_weights

    plugin = GeminiPlugin(
        max_norm=1.0,
        initial_scale=2**5,
        master_weights=master_weights,
        enable_gradient_accumulation=enable_gradient_accumulation,
        **placement_config,
    )
    booster = Booster(plugin=plugin)

    lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
    model = booster.enable_lora(model, lora_config=lora_config)
    model_copy = copy.deepcopy(model)

    optimizer = HybridAdam(model.parameters(), lr=0.001)
    criterion = loss_fn

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    if enable_gradient_accumulation:
        # Do forward and backward under grad accum setting.
        accum_iter = 2
        for i in range(2 * accum_iter):
            data = data_gen_fn()
            data = {
                k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v
                for k, v in data.items()
            }
            output = model(**data)
            output = output_transform_fn(output)
            loss = criterion(output) / accum_iter
            booster.backward(loss, optimizer)

            if (i + 1) % accum_iter == 0:
                optimizer.clip_grad_by_norm(1.0)
                optimizer.step()
    else:
        do_fwd_bwd(booster, model, optimizer, data_gen_fn, output_transform_fn, criterion)

    # Check parameters
    gemini_dict = model.state_dict(only_rank_0=False)
    model_copy_dict = model_copy.state_dict()

    for name in gemini_dict:
        check_param_equality(
            name, gemini_dict[name], model_copy_dict[name], modules_to_save=model.unwrap().modules_to_save
        )

    # check the checkpointio function
    with shared_tempdir() as tempdir:
        lora_ckpt_path = os.path.join(tempdir, "model_ckpt")
        optimizer_ckpt_path = os.path.join(tempdir, "optimizer_ckpt")

        booster.save_lora_as_pretrained(model, lora_ckpt_path)
        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=False)
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

        check_state_dict_equal(model.state_dict(only_rank_0=False), model_load.state_dict(only_rank_0=False))
        check_state_dict_equal(optimizer.state_dict(only_rank_0=False), optimizer_load.state_dict(only_rank_0=False))


@parameterize("placement_config", PLACEMENT_CONFIGS)
@parameterize("master_weights", [True, False])
def run_lora_test(placement_config: dict, master_weights: bool):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        task_type = None
        if name == "transformers_llama_for_casual_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"

        check_fn(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type, placement_config, master_weights)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_lora_test()


@rerun_if_address_is_in_use()
def test_gemini_lora():
    spawn(run_dist, 2)
