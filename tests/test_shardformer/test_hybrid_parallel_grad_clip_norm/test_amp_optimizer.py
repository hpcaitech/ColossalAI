import pytest
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

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
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    bert = unwrap_model(org_model, "BertModel", "bert")
    sharded_bert = unwrap_model(sharded_model, "BertModel", "bert")

    col_layer_for_check = ["encoder.layer[0].output.dense"]
    row_layer_for_check = ["embeddings.word_embeddings", "encoder.layer[0].intermediate.dense"]

    if test_config["precision"] == "fp32":
        atol, rtol = 1e-4, 1e-3
    elif test_config["precision"] == "fp16":
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 2e-2, 2e-2

    # Check grads
    # Save gradient tensors for comparison between the original model and the sharded model.
    grads_to_check = {}
    if (stage_manager is None or stage_manager.is_first_stage()) and booster.plugin.zero_stage == 0:
        col_layer_grads = get_grad_tensors_for_check(
            bert, sharded_bert, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False
        )
        row_layer_grads = get_grad_tensors_for_check(
            bert, sharded_bert, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0, verbose=False
        )
        grads_to_check.update(col_layer_grads)
        grads_to_check.update(row_layer_grads)
    check_all_grad_tensors(grads_to_check)

    # Check gradient norm
    # Convert the gradient data of the working parameter to float and assign it to the master parameter's gradient
    # Note that this operation should have been done in the 'step' function, but it is performed here in advance for gradient norm calculation purposes.
    # Although it will be done again in the 'step' function, it does not affect correctness.
    for group in sharded_optimizer.optim.param_groups:
        for p in group["params"]:
            working_param = sharded_optimizer.master_to_working_map[p]
            if p is working_param:
                continue
            if working_param.grad is not None:
                p.grad = working_param.grad.data.float()
                working_param.grad = None
    # Create a list of parameter-gradient pairs containing working parameters and their gradients
    param_gradient_pairs = [
        (sharded_optimizer.master_to_working_map[p], p.grad)
        for group in sharded_optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]

    origin_norm = clip_grad_norm_(org_model.parameters(), test_config["max_norm"])
    # Calculate the gradient norm of the sharded optimizer
    device = origin_norm.device
    hybrid_norm = torch.tensor(sharded_optimizer._compute_grad_norm(param_gradient_pairs)).to(device)

    # If using fp16 precision, divide by the initial scale
    if test_config["precision"] == "fp16":
        hybrid_norm /= test_config["initial_scale"]

    # Assert that the gradient norm of the original model is close to the gradient norm of the hybrid model
    assert torch.allclose(
        origin_norm, hybrid_norm, atol=atol, rtol=rtol
    ), f"Original model grad norm is not equal to sharded model grad norm\n{origin_norm}\n{hybrid_norm}"

    # Optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # Check last hidden state & loss
    if stage_manager is None or stage_manager.is_last_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-5, 1e-3
        elif test_config["precision"] == "fp16":
            atol, rtol = 5e-3, 5e-3
        else:
            atol, rtol = 2e-2, 2e-2
        if org_model.__class__.__name__ == "BertModel":
            check_output_hidden_state(org_output, sharded_output, stage_manager, atol=atol, rtol=rtol)

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

    # Check weights
    if test_config["precision"] == "fp32":
        atol, rtol = 5e-3, 1e-3
    else:
        atol, rtol = 5e-3, 5e-3
    if stage_manager is None or stage_manager.is_first_stage():
        check_weight(bert, sharded_bert, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False)

    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": True,
            "use_lazy_init": False,
            "precision": "fp16",
            "max_norm": 5,
            "initial_scale": 1,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": True,
            "use_lazy_init": False,
            "precision": "bf16",
            "max_norm": 5,
        },
    ],
)
def run_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
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
            "enable_all_optimization": True,
            "use_lazy_init": False,
            "precision": "bf16",
            "max_norm": 5,
        },
        {
            "tp_size": 2,
            "pp_size": 2,
            "num_microbatches": 4,
            "enable_all_optimization": True,
            "use_lazy_init": False,
            "precision": "fp16",
            "max_norm": 5,
            "initial_scale": 1,
        },
    ],
)
def run_3d_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def check_grad_clip_norm(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_test()


def check_grad_clip_norm_3d(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_3d_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_grad_clip_norm():
    spawn(check_grad_clip_norm, 4)


@pytest.mark.largedist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_grad_clip_norm_3d():
    spawn(check_grad_clip_norm_3d, 8)


if __name__ == "__main__":
    test_grad_clip_norm()
    test_grad_clip_norm_3d()
