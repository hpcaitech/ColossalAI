import copy
import os
from itertools import product

import torch
from peft import LoraConfig
from torch import distributed as dist
from torch.optim import AdamW

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.testing import check_state_dict_equal, clear_cache_before_run, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_checkpoint_io.utils import shared_tempdir


@clear_cache_before_run()
def check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type):
    model = model_fn()
    lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)

    test_plugins = [TorchDDPPlugin(), LowLevelZeroPlugin(), HybridParallelPlugin(tp_size=1, pp_size=1)]
    test_configs = [
        {
            "lora_config": lora_config,
            "quantize": False,
        },
        {
            "lora_config": lora_config,
            "quantize": True,
        },
    ]
    for plugin, test_config in product(test_plugins, test_configs):
        # checkpoint loaded model
        model_save = model_fn()
        model_load = copy.deepcopy(model_save)

        optimizer = AdamW(model.parameters(), lr=0.001)
        criterion = loss_fn

        booster = Booster(plugin=plugin)
        model_save = booster.enable_lora(model_save, **test_config)
        model_save, optimizer, criterion, _, _ = booster.boost(model_save, optimizer, criterion)

        with shared_tempdir() as tempdir:
            lora_ckpt_path = os.path.join(tempdir, "ckpt")
            booster.save_lora_as_pretrained(model_save, lora_ckpt_path)
            dist.barrier()

            # The Lora checkpoint should be small in size
            checkpoint_size_mb = os.path.getsize(os.path.join(lora_ckpt_path, "adapter_model.bin")) / (1024 * 1024)
            assert checkpoint_size_mb < 1

            model_load = booster.enable_lora(model_load, pretrained_dir=lora_ckpt_path, **test_config)
            model_load, _, _, _, _ = booster.boost(model_load)

            check_state_dict_equal(model_save.state_dict(), model_load.state_dict())

        # test fwd bwd correctness
        test_model = model_load
        if isinstance(model_load, HybridParallelModule):
            model_load = model_load.module.module
        model_copy = copy.deepcopy(model_load)

        data = data_gen_fn()
        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        output = test_model(**data)
        output = output_transform_fn(output)
        loss = criterion(output)

        booster.backward(loss, optimizer)
        optimizer.clip_grad_by_norm(1.0)
        optimizer.step()

        for (n1, p1), (n2, p2) in zip(test_model.named_parameters(), model_copy.named_parameters()):
            if "lora_" in n1:
                # lora modules require gradients, thus updated
                assert p1.requires_grad
                assert not torch.testing.assert_close(p1.to(p2.device).to(p2.dtype), p2, atol=5e-3, rtol=5e-3)
            else:
                if not p1.requires_grad:
                    torch.testing.assert_close(p1.to(p2.device).to(p2.dtype), p2, atol=5e-3, rtol=5e-3)


def run_lora_test():
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        task_type = None
        if name == "transformers_llama_for_causal_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"
        check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_lora_test()


@rerun_if_address_is_in_use()
def test_torch_ddp_lora():
    spawn(run_dist, 2)
