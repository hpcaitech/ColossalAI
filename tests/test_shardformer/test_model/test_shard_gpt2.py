import pytest
import torch
import torch.distributed as dist

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


def print_rank(prompt, value, rank=0):
    if dist.get_rank() == rank:
        print(f"rank-{rank}, {prompt}: {value}")


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config
    )

    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # unwrap model
    gpt2 = unwrap_model(org_model, "GPT2Model", "transformer")
    sharded_gpt2 = unwrap_model(sharded_model, "GPT2Model", "transformer")

    norm_layer_for_check = ["h[0].ln_1", "h[0].ln_2"]
    col_layer_for_check = ["h[0].mlp.c_fc"]
    row_layer_for_check = ["wte", "h[0].mlp.c_proj"]

    # Save gradient tensors for comparison between the original model and the sharded model.
    grads_to_check = {}
    if (stage_manager is None or stage_manager.is_first_stage()) and booster.plugin.zero_stage == 0:
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        col_layer_grads = get_grad_tensors_for_check(
            gpt2, sharded_gpt2, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False
        )
        row_layer_grads = get_grad_tensors_for_check(
            gpt2, sharded_gpt2, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0, verbose=False
        )

        norm_layer_grads = get_grad_tensors_for_check(
            gpt2,
            sharded_gpt2,
            norm_layer_for_check,
            tp_group,
            atol=atol,
            rtol=rtol,
            dim=1,
            verbose=False,
        )

        grads_to_check.update(col_layer_grads)
        grads_to_check.update(row_layer_grads)
        grads_to_check.update(norm_layer_grads)

    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # check last hidden state & loss
    if stage_manager is None or stage_manager.is_last_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-5, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3

        if org_model.__class__.__name__ == "GPT2Model":
            check_output_hidden_state(org_output, sharded_output, stage_manager, atol=atol, rtol=rtol)

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

    # check weights
    if stage_manager is None or stage_manager.is_first_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 5e-3, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        check_weight(gpt2, sharded_gpt2, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False)

    # check grads
    check_all_grad_tensors(grads_to_check)

    Randomizer.reset_index()
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 4,
            "pp_size": 1,
            "num_microbatches": 1,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "2",
            "enable_flash_attention": True,
            "use_lazy_init": True,
            "precision": "fp32",
            "initial_scale": 1,
        },
        # {
        #     "tp_size": 1,
        #     "pp_size": 1,
        #     "num_microbatches": 1,
        #     "enable_sequence_parallelism": True,
        #     "sequence_parallelism_mode": "3",
        #     "enable_flash_attention": True,
        #     "use_lazy_init": True,
        #     "precision": "fp32",
        #     "initial_scale": 1,
        # },
        # {
        #     "tp_size": 2,
        #     "pp_size": 2,
        #     "num_microbatches": 4,
        #     "enable_all_optimization": True,
        #     "use_lazy_init": True,
        #     "precision": "fp16",
        #     "initial_scale": 1,
        # },
        # {
        #     "tp_size": 1,
        #     "pp_size": 2,
        #     "num_microbatches": 4,
        #     "enable_all_optimization": True,
        #     "use_lazy_init": True,
        #     "precision": "fp16",
        #     "initial_scale": 1,
        # },
        # {
        #     "tp_size": 4,
        #     "pp_size": 1,
        #     "enable_all_optimization": False,
        #     "enable_flash_attention": True,
        #     "use_lazy_init": False,
        #     "precision": "fp32",
        # },
        # {
        #     "tp_size": 2,
        #     "pp_size": 1,
        #     "enable_all_optimization": True,
        #     "use_lazy_init": False,
        #     "precision": "fp32",
        # },
        # {
        #     "tp_size": 2,
        #     "pp_size": 2,
        #     "num_microbatches": 4,
        #     "enable_all_optimization": True,
        #     "use_lazy_init": True,
        #     "precision": "fp32",
        # },
        # {
        #     "tp_size": 2,
        #     "pp_size": 1,
        #     "enable_all_optimization": True,
        #     "use_lazy_init": True,
        #     "zero_stage": 2,
        #     "precision": "fp16",
        #     "initial_scale": 1,
        # },
        # {
        #     "tp_size": 1,
        #     "pp_size": 2,
        #     "num_microbatches": 2,
        #     "enable_all_optimization": True,
        #     "use_lazy_init": True,
        #     "zero_stage": 1,
        #     "precision": "fp16",
        #     "initial_scale": 1,
        # },
    ],
)
@clear_cache_before_run()
def run_gpt2_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_gpt", exclude="transformers_gptj")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name in [
            # 'transformers_gpt',
            "transformers_gpt_lm",  # grad mismatch
            "transformers_gpt_double_heads",  # grad mismatch
            "transformers_gpt_for_question_answering",
            "transformers_gpt_for_token_classification",
            "transformers_gpt_for_sequence_classification",
        ]:
            continue
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp32",
            "initial_scale": 1,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": False,
            "use_lazy_init": False,
            "precision": "fp16",
            "zero_stage": 1,
            "initial_scale": 1,
        },
    ],
)
@clear_cache_before_run()
def run_gpt2_3d_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_gpt", exclude="transformers_gptj")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


def check_gpt2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_gpt2_test()


def check_gpt2_3d(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_gpt2_3d_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_gpt2():
    spawn(check_gpt2, 4)


@pytest.mark.largedist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_gpt2_3d():
    spawn(check_gpt2_3d, 8)


if __name__ == "__main__":
    test_gpt2()
    test_gpt2_3d()
