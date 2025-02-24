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
    unwrap_model,
)


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config
    )

    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model,
        sharded_model,
        sharded_optimizer,
        data_gen_fn,
        output_transform_fn,
        criterion,
        booster,
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # unwrap model
    t5 = unwrap_model(org_model)
    sharded_t5 = unwrap_model(sharded_model)

    if t5.__class__.__name__ == "T5ForTokenClassification":
        row_layer_for_check = ["transformer.shared", "transformer.encoder.block[0].layer[0].SelfAttention.q"]
    else:
        row_layer_for_check = ["shared", "encoder.block[0].layer[0].SelfAttention.q"]

    # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
    grads_to_check = {}
    if test_config["precision"] == "fp32":
        atol, rtol = 1e-5, 1e-3
    else:
        atol, rtol = 9e-2, 0
    if (stage_manager is None or stage_manager.is_first_stage()) and booster.plugin.zero_stage == 0:
        row_layer_grads = get_grad_tensors_for_check(
            t5, sharded_t5, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0
        )
        grads_to_check.update(row_layer_grads)

    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # check last hidden state & loss
    if stage_manager is None or stage_manager.is_last_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-5, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3

        if org_model.__class__.__name__ not in ["T5ForConditionalGeneration", "T5ForTokenClassification"]:
            check_output_hidden_state(org_output, sharded_output, stage_manager, atol=atol, rtol=rtol)

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

    # check weights
    if test_config["precision"] == "fp32":
        # TODO he precision in weight checking is too significant.
        atol, rtol = 1e-3, 1e-3
    else:
        atol, rtol = 5e-3, 5e-3
    if stage_manager is None or stage_manager.is_first_stage():
        check_weight(
            t5,
            sharded_t5,
            row_layer_for_check,
            tp_group,
            atol=atol,
            rtol=rtol,
            dim=0,
            verbose=False,
        )

    # check grads
    check_all_grad_tensors(grads_to_check)

    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 2,
            "enable_metadata_cache": False,
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {
            "tp_size": 1,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_metadata_cache": False,
            "use_lazy_init": False,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {
            "tp_size": 4,
            "pp_size": 1,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp32",
        },
        {
            "tp_size": 1,
            "pp_size": 4,
            "num_microbatches": 4,
            "enable_metadata_cache": False,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp32",
        },
        {
            "tp_size": 2,
            "pp_size": 1,
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "zero_stage": 2,
            "precision": "fp16",
            "initial_scale": 1,
        },
        {
            "tp_size": 1,
            "pp_size": 2,
            "num_microbatches": 2,
            "enable_metadata_cache": False,
            "enable_all_optimization": True,
            "use_lazy_init": True,
            "zero_stage": 1,
            "precision": "fp16",
            "initial_scale": 1,
        },
    ],
)
@clear_cache_before_run()
def run_t5_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry(["transformers_t5_for_token_classification"])

    for name, (
        model_fn,
        data_gen_fn,
        output_transform_fn,
        loss_fn,
        _,
    ) in sub_model_zoo.items():
        # skip 4-stage pp test for t5_encoder
        if test_config["pp_size"] > 2 and name in [
            "transformers_t5_encoder_model",
            "transformers_t5_for_token_classification",
        ]:
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
            "num_microbatches": 4,
            "enable_metadata_cache": False,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp16",
            "zero_stage": 1,
            "initial_scale": 1,
        },
    ],
)
def run_t5_3d_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_t5")

    for name, (
        model_fn,
        data_gen_fn,
        output_transform_fn,
        loss_fn,
        _,
    ) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


def check_t5(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    run_t5_test()


def check_t5_3d(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    run_t5_3d_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_t5():
    spawn(check_t5, 4)


@pytest.mark.largedist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_t5_3d():
    spawn(check_t5_3d, 8)


if __name__ == "__main__":
    test_t5()
    test_t5_3d()
