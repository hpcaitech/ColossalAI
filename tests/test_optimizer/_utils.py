import pytest
import torch
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)


def check_bert_fwd_bwd(
    model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config, optim_class, sharded_optim_class
):
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config, optim_class, sharded_optim_class
    )

    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    bert = unwrap_model(org_model, "BertModel", "bert")
    sharded_bert = unwrap_model(sharded_model, "BertModel", "bert")
    weight_layer_for_check = ["encoder.layer[0].output.dense", "encoder.layer[1].output.dense"]

    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()

    # check weights
    if test_config["precision"] == "bf16":
        atol, rtol = 5e-4, 1e-4
    else:
        atol, rtol = 5e-4, 5e-4
    if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
        check_weight(bert, sharded_bert, weight_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1)

    # check optim states
    for group in org_optimizer.param_groups:
        for p in group["params"]:
            sharded_state = sharded_optimizer.optim.state[p]
            state = org_optimizer.state[p]
            for key in sharded_state:
                assert_close(state[key], sharded_state[key], rtol=1e-5, atol=1e-5)
    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        {
            "tp_size": 1,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "bf16",
        },
        {
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "bf16",
        },
        {
            "tp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "bf16",
        },
        {
            "tp_size": 1,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "fp16",
        },
        {
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "fp16",
        },
        {
            "tp_size": 4,
            "num_microbatches": 4,
            "zero_stage": 2,
            "precision": "fp16",
        },
    ],
)
def run_bert_test(test_config, optim_class, sharded_optim_class):
    """Just call this if you've initialized distributed backend and spawned procs"""
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["pp_size"] = 1  # Do NOT test Pipeline Parallel
    test_config["initial_scale"] = 2**15  # avoid overflow

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_bert_fwd_bwd(
            model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config, optim_class, sharded_optim_class
        )

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def _run_bert_test(rank, world_size, port, optim_class, sharded_optim_class):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_bert_test(optim_class, sharded_optim_class)


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def check_optim_on_bert(optim_class, sharded_optim_class):
    spawn(_run_bert_test, 4, optim_class, sharded_optim_class)
