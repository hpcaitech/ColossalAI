from copy import deepcopy
from typing import Optional

import torch
import torch.distributed as dist
from peft import LoraConfig
from torchvision.models import resnet18
from utils import shared_tempdir

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    check_state_dict_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from colossalai.zero import LowLevelZeroOptimizer
from tests.kit.model_zoo import model_zoo


# stage 1 and 2 process the optimizer/mode the same way
# only test 2 is fine
@clear_cache_before_run()
@parameterize("stage", [2])
@parameterize("shard", [True, False])
@parameterize("offload", [False, True])
def check_low_level_zero_checkpointIO(stage: int, shard: bool, offload: bool):
    plugin = LowLevelZeroPlugin(stage=stage, max_norm=1.0, initial_scale=32, cpu_offload=offload)
    booster = Booster(plugin=plugin)
    model = resnet18()
    criterion = lambda x: x.mean()
    optimizer = HybridAdam((model.parameters()), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    x = torch.randn(1, 3, 224, 224, device="cuda")
    output = model(x)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.step()
    with shared_tempdir() as tempdir:
        model_ckpt_path = f"{tempdir}/model"
        optimizer_ckpt_path = f"{tempdir}/optimizer"
        # lr scheduler is tested in test_torch_ddp_checkpoint_io.py and low level zero does not change it, we can skip it here
        booster.save_model(model, model_ckpt_path, shard=shard)
        booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=shard)

        dist.barrier()

        new_model = resnet18()
        new_optimizer = HybridAdam((new_model.parameters()), lr=0.001)
        new_model, new_optimizer, _, _, _ = booster.boost(new_model, new_optimizer)

        booster.load_model(new_model, model_ckpt_path)
        check_state_dict_equal(model.state_dict(), new_model.state_dict())
        # check master weight
        assert isinstance(new_optimizer, LowLevelZeroOptimizer)
        working_param_id_set = set(id(p) for p in new_model.parameters())
        for p_id, master_param in new_optimizer.working_to_master_param.items():
            assert p_id in working_param_id_set
            working_param = new_optimizer.master_to_working_param[id(master_param)]
            padding = new_optimizer.get_param_padding_size(working_param)
            padded_param = torch.nn.functional.pad(working_param.data.view(-1), (0, padding))
            working_shard = padded_param.chunk(dist.get_world_size())[dist.get_rank()]
            assert torch.equal(
                working_shard, master_param.data.view(-1).to(dtype=padded_param.dtype, device=padded_param.device)
            )

        booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
        check_state_dict_equal(optimizer.optim.state_dict(), new_optimizer.optim.state_dict())
    torch.cuda.empty_cache()


def run_fn(stage, shard, offload, model_fn, data_gen_fn, output_transform_fn, lora_config=None) -> Optional[str]:
    try:
        plugin = LowLevelZeroPlugin(stage=stage, max_norm=1.0, initial_scale=2**5, cpu_offload=offload)
        new_plugin = LowLevelZeroPlugin(stage=stage, max_norm=1.0, initial_scale=2**5, cpu_offload=offload)
        booster = Booster(plugin=plugin)
        new_booster = Booster(plugin=new_plugin)
        model = model_fn()
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        new_model = deepcopy(model)
        new_optimizer = HybridAdam(new_model.parameters(), lr=1e-3)
        model = booster.enable_lora(model, lora_config=lora_config)
        criterion = lambda x: x.mean()
        data = data_gen_fn()

        data = {
            k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

        output = model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])

        booster.backward(loss, optimizer)
        optimizer.step()

        with shared_tempdir() as tempdir:
            model_ckpt_path = f"{tempdir}/model"
            optimizer_ckpt_path = f"{tempdir}/optimizer"

            booster.save_lora_as_pretrained(model, model_ckpt_path)
            booster.save_optimizer(optimizer, optimizer_ckpt_path, shard=False)
            new_model = new_booster.enable_lora(new_model, pretrained_dir=model_ckpt_path, lora_config=lora_config)
            new_model, new_optimizer, criterion, _, _ = new_booster.boost(new_model, new_optimizer, criterion)
            check_state_dict_equal(model.state_dict(), new_model.state_dict())

            # check master weight
            assert isinstance(new_optimizer, LowLevelZeroOptimizer)
            working_param_id_set = set(id(p) for p in new_model.parameters())
            for p_id, master_param in new_optimizer.working_to_master_param.items():
                assert p_id in working_param_id_set
                working_param = new_optimizer.master_to_working_param[id(master_param)]
                padding = new_optimizer.get_param_padding_size(working_param)
                padded_param = torch.nn.functional.pad(working_param.data.view(-1), (0, padding))
                working_shard = padded_param.chunk(dist.get_world_size())[dist.get_rank()]
                assert torch.equal(
                    working_shard, master_param.data.view(-1).to(dtype=padded_param.dtype, device=padded_param.device)
                )

            new_booster.load_optimizer(new_optimizer, optimizer_ckpt_path)
            check_state_dict_equal(optimizer.optim.state_dict(), new_optimizer.optim.state_dict())

    except Exception as e:
        # return repr(e)
        raise e


@clear_cache_before_run()
@parameterize("stage", [2])
@parameterize("shard", [True, False])
@parameterize("offload", [False, True])
@parameterize("model_name", ["transformers_llama"])
def check_low_level_zero_lora_checkpointIO(
    stage: int, shard: bool, offload: bool, model_name: str, early_stop: bool = True
):
    passed_models = []
    failed_info = {}  # (model_name, error) pair

    sub_model_zoo = model_zoo.get_sub_registry(model_name)
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name != "transformers_llama":
            continue
        task_type = None
        if name == "transformers_llama_for_causal_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"
        lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
        err = run_fn(stage, shard, offload, model_fn, data_gen_fn, output_transform_fn, lora_config)

        torch.cuda.empty_cache()

        if err is None:
            passed_models.append(name)
        else:
            failed_info[name] = err
            if early_stop:
                break

    if dist.get_rank() == 0:
        print(f"Passed models({len(passed_models)}): {passed_models}\n\n")
        print(f"Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n")
    assert len(failed_info) == 0, "\n".join([f"{k}: {v}" for k, v in failed_info.items()])


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_low_level_zero_checkpointIO()
    check_low_level_zero_lora_checkpointIO()
    torch.cuda.empty_cache()


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_low_level_zero_checkpointIO():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_low_level_zero_checkpointIO()
