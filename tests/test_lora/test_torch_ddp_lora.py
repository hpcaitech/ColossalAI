import copy
import os

from peft import LoraConfig
from torch import distributed as dist
from torch.optim import AdamW

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.testing import check_state_dict_equal, clear_cache_before_run, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_checkpoint_io.utils import shared_tempdir
from tests.test_lora.utils import check_param_equality, do_fwd_bwd


@clear_cache_before_run()
def check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type):
    model = model_fn()
    lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)

    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    model = booster.enable_lora(model, lora_config=lora_config)
    model_copy = copy.deepcopy(model)

    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = loss_fn

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)
    do_fwd_bwd(booster, model, optimizer, data_gen_fn, output_transform_fn, criterion)

    for (n1, p1), (_, p2) in zip(model.named_parameters(), model_copy.named_parameters()):
        check_param_equality(n1, p1, p2, modules_to_save=model.unwrap().modules_to_save)


@clear_cache_before_run()
def check_checkpoint(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type):
    plugin = TorchDDPPlugin()
    lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
    criterion = loss_fn

    model_save = model_fn()
    model_load = copy.deepcopy(model_save)

    booster = Booster(plugin=plugin)
    model_save = booster.enable_lora(model_save, lora_config=lora_config)
    optimizer_save = AdamW(model_save.parameters(), lr=0.001)

    model_save, optimizer_save, _, _, _ = booster.boost(model_save, optimizer_save)
    do_fwd_bwd(booster, model_save, optimizer_save, data_gen_fn, output_transform_fn, criterion)

    with shared_tempdir() as tempdir:
        lora_ckpt_path = os.path.join(tempdir, "model_ckpt")
        optimizer_ckpt_path = os.path.join(tempdir, "optimizer_ckpt")

        booster.save_lora_as_pretrained(model_save, lora_ckpt_path)
        booster.save_optimizer(optimizer_save, optimizer_ckpt_path)
        dist.barrier()

        # The Lora checkpoint should be small in size
        model_checkpoint_size_mb = os.path.getsize(os.path.join(lora_ckpt_path, "adapter_model.bin")) / (1024 * 1024)
        optimizer_checkpoint_size_mb = os.path.getsize(optimizer_ckpt_path) / (1024 * 1024)
        assert model_checkpoint_size_mb < 1 and optimizer_checkpoint_size_mb < 1

        model_load = booster.enable_lora(model_load, pretrained_dir=lora_ckpt_path)
        optimizer_load = AdamW(model_save.parameters(), lr=0.001)
        model_load, optimizer_load, _, _, _ = booster.boost(model_load, optimizer_load)

        booster.load_optimizer(optimizer_load, optimizer_ckpt_path)

        check_state_dict_equal(model_save.state_dict(), model_load.state_dict())
        check_state_dict_equal(optimizer_save.state_dict(), optimizer_load.state_dict())


def run_lora_test():
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        task_type = None
        if name == "transformers_llama_for_casual_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"
        check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type)
        check_checkpoint(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_lora_test()


@rerun_if_address_is_in_use()
def test_torch_ddp_lora():
    spawn(run_dist, 2)
