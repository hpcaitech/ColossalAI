import pytest
import torch

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    get_grad_tensors_for_check,
    run_forward_backward_with_hybrid_plugin,
)


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    # check forward
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config
    )

    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # unwarp the model
    if org_model.__class__.__name__ == "WhisperForConditionalGeneration":
        whisper = org_model.model
        sharded_whisper = sharded_model.unwrap().model
    else:
        whisper = org_model
        sharded_whisper = sharded_model.unwrap()

    # check grad
    if org_model.__class__.__name__ == "WhisperForAudioClassification":
        col_layer_for_check = ["encoder.layers[0].self_attn.q_proj"]
        row_layer_for_check = ["encoder.layers[0].self_attn.out_proj"]
    else:
        col_layer_for_check = [
            "encoder.layers[0].self_attn.q_proj",
            # 'decoder.layers[0].self_attn.q_proj'
        ]
        row_layer_for_check = [
            "encoder.layers[0].self_attn.out_proj",
            #'decoder.layers[0].self_attn.out_proj'
        ]

    # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
    grads_to_check = {}
    if test_config["precision"] == "fp32":
        atol, rtol = 2e-4, 2e-4
    else:
        atol, rtol = 5e-3, 5e-3

    if stage_manager is None or stage_manager.is_first_stage():
        row_layer_grads = get_grad_tensors_for_check(
            whisper, sharded_whisper, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1
        )
        col_layer_grads = get_grad_tensors_for_check(
            whisper, sharded_whisper, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0
        )
        grads_to_check.update(col_layer_grads)
        grads_to_check.update(row_layer_grads)

    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # check last hidden state & loss
    if stage_manager is None or stage_manager.is_last_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 2e-4, 2e-4
        else:
            atol, rtol = 5e-3, 5e-3

        if org_model.__class__.__name__ == "WhisperModel":
            check_output_hidden_state(org_output, sharded_output, stage_manager, atol=atol, rtol=rtol)

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

    # check weights
    if test_config["precision"] == "fp32":
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 5e-3, 5e-3
    if stage_manager is None or stage_manager.is_first_stage():
        check_weight(
            whisper, sharded_whisper, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False
        )
        check_weight(
            whisper, sharded_whisper, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0, verbose=False
        )

    # check grads
    check_all_grad_tensors(grads_to_check)

    torch.cuda.empty_cache()


# TODO fix WhisperForConditionalGeneration enable jit fused operato
# TODO（jianghai) fix fp16
@parameterize(
    "test_config",
    [
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 2,
            "enable_metadata_cache": False,
            "enable_all_optimization": True,
            "use_lazy_init": False,
            "precision": "fp32",
            "initial_scale": 1,
        },
        {
            "tp_size": 1,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_metadata_cache": False,
            "use_lazy_init": False,
            "precision": "fp32",
            "initial_scale": 1,
        },
        {
            "tp_size": 4,
            "pp_size": 1,
            "enable_all_optimization": True,
            "use_lazy_init": False,
            "precision": "fp32",
        },
        {
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "enable_metadata_cache": False,
            "use_lazy_init": False,
            "precision": "fp32",
        },
        # whisper is not supported fp16 for now.
    ],
)
def run_whisper_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_whisper")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if test_config["pp_size"] > 2 and name == "transformers_whisper_for_audio_classification":
            continue
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_metadata_cache": False,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp32",
            "initial_scale": 1,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 2,
            "enable_metadata_cache": False,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp32",
            "initial_scale": 1,
        },
    ],
)
def run_whisper_3d_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_whisper")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


def check_whisper(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_whisper_test()


def check_whisper_3d(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_whisper_3d_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_whisper():
    spawn(check_whisper, 4)


@pytest.mark.largedist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_whisper_3d():
    spawn(check_whisper_3d, 8)


if __name__ == "__main__":
    test_whisper()
    test_whisper_3d()
