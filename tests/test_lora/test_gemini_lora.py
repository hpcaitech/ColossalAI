import copy

import torch
from peft import LoraConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import (
    assert_equal,
    assert_not_equal,
    clear_cache_before_run,
    parameterize,
    rerun_if_address_is_in_use,
    spawn,
)
from tests.kit.model_zoo import model_zoo

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

    # for n, p in model.named_parameters():
    #     print(n, p.shape, p.requires_grad)

    data = data_gen_fn()
    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}

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
            # if a non-lora module isn't supposed to be saved, it shouldn't be updated
            modules_to_save = model.unwrap().modules_to_save
            if (modules_to_save is None) or all((key not in name) for key in modules_to_save):
                assert not model.name2param[name].requires_grad
                assert_equal(p1, p2)


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
        # check_checkpoint()
        # check_grad_accum()


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_lora_test()


@rerun_if_address_is_in_use()
def test_gemini_lora():
    spawn(run_dist, 2)
