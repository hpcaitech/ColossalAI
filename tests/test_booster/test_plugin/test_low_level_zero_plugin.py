from typing import Optional

import torch
import torch.distributed as dist
from peft import LoraConfig
from torch.optim import Adam

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin

# from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import COMMON_MODELS, IS_FAST_TEST, model_zoo

# These models are not compatible with AMP
_AMP_ERR_MODELS = ["timm_convit", "deepfm_interactionarch"]
# These models have no parameters
_LOW_LEVEL_ZERO_ERR_MODELS = ["dlrm_interactionarch"]
# These models will cause stuck, to be fixed
_STUCK_MODELS = ["transformers_albert_for_multiple_choice"]


@clear_cache_before_run()
def run_fn(stage, model_fn, data_gen_fn, output_transform_fn, lora_config=None) -> Optional[str]:
    device = get_accelerator().get_current_device()
    try:
        plugin = LowLevelZeroPlugin(stage=stage, max_norm=1.0, initial_scale=2**5)
        booster = Booster(plugin=plugin)
        model = model_fn()
        optimizer = Adam(model.parameters(), lr=1e-3)

        if lora_config is not None:
            model = booster.enable_lora(model, lora_config=lora_config)

        criterion = lambda x: x.mean()
        data = data_gen_fn()

        data = {
            k: v.to(device) if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()
        }

        model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

        output = model(**data)
        output = output_transform_fn(output)
        output_key = list(output.keys())[0]
        loss = criterion(output[output_key])

        booster.backward(loss, optimizer)
        optimizer.step()

    except Exception as e:
        return repr(e)
        # raise e


@parameterize("stage", [2])
def check_low_level_zero_plugin(stage: int, early_stop: bool = True):
    """check low level zero plugin over model zoo

    Args:
        stage (int), stage of low level zero plugin
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    passed_models = []
    failed_info = {}  # (model_name, error) pair
    ignore_models = _AMP_ERR_MODELS + _LOW_LEVEL_ZERO_ERR_MODELS + _STUCK_MODELS
    skipped_models = []

    if IS_FAST_TEST:
        registry = model_zoo.get_sub_registry(COMMON_MODELS)
    else:
        registry = model_zoo

    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in registry.items():
        # FIXME(ver217): fix these models
        if name in ignore_models:
            skipped_models.append(name)
            continue
        err = run_fn(stage, model_fn, data_gen_fn, output_transform_fn)
        get_accelerator().empty_cache()

        if err is None:
            passed_models.append(name)
        else:
            failed_info[name] = err
            if early_stop:
                break

    if dist.get_rank() == 0:
        print(f"Passed models({len(passed_models)}): {passed_models}\n\n")
        print(f"Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n")
        print(f"Skipped models({len(skipped_models)}): {skipped_models}\n\n")
    assert len(failed_info) == 0, "\n".join([f"{k}: {v}" for k, v in failed_info.items()])


@parameterize("stage", [2])
@parameterize("model_name", ["transformers_llama"])
def check_low_level_zero_lora(stage, model_name, early_stop: bool = True):
    passed_models = []
    failed_info = {}  # (model_name, error) pair

    sub_model_zoo = model_zoo.get_sub_registry(model_name)
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        task_type = None
        if name == "transformers_llama_for_causal_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"
        lora_config = LoraConfig(task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
        err = run_fn(stage, model_fn, data_gen_fn, output_transform_fn, lora_config)

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


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_low_level_zero_plugin(early_stop=early_stop)
    check_low_level_zero_lora(early_stop=early_stop)


@rerun_if_address_is_in_use()
def test_low_level_zero_plugin(early_stop: bool = True):
    spawn(run_dist, 2, early_stop=early_stop)


if __name__ == "__main__":
    test_low_level_zero_plugin(early_stop=False)
