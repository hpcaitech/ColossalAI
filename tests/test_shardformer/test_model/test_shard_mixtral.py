# modified from test_shard_mistral.py
import os

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
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

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):
    # TODO: SGD failed for full dp
    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
        model_fn, loss_fn, test_config, pluggin_cls=MoeHybridParallelPlugin, optim_class=torch.optim.SGD
    )

    org_model = org_model.to(torch.float16)
    org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
        org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
    )
    print(org_output.last_hidden_state.shape, sharded_output.last_hidden_state.shape)
    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # check last hidden state & loss
    if stage_manager is None or stage_manager.is_last_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 1e-5, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)
        check_output_hidden_state(org_output, sharded_output, stage_manager, atol, rtol)

    # unwrap model
    mixtral_model = unwrap_model(org_model, "MixtralModel", "model")
    shard_mixtral_model = unwrap_model(sharded_model, "MixtralModel", "model")

    row_layer_for_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
    col_layer_for_check = ["layers[0].self_attn.o_proj"]

    # Check the grad when using ZeRO-1 and ZeRO-2
    if (
        # booster.plugin.zero_stage in [1, 2]
        booster.plugin.shard_config.enable_sequence_parallelism
        and booster.plugin.shard_config.sequence_parallelism_mode == "all_to_all"
    ):
        rank = dist.get_rank()
        # for p1, p2 in zip(mixtral_model.parameters(), sharded_optimizer._master_param_groups_of_current_rank[0]):
        for (n1, p1), (n2, p2) in zip(mixtral_model.named_parameters(), shard_mixtral_model.named_parameters()):
            try:
                assert_close(p1.grad, p2.grad, atol=5e-3, rtol=5e-3, check_dtype=False)
                print(f"{rank=},passed grad: {n1}, {n2}")
            except Exception as e:
                print(f"{rank=},failed grad: {n1} {p1.grad[:2,:2]}, {n2} {p2.grad[:2, :2]}")
                raise e

    # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
    grads_to_check = {}
    if (stage_manager is None or stage_manager.is_first_stage()) and booster.plugin.zero_stage == 0:
        if test_config["precision"] == "fp32":
            atol, rtol = 5e-5, 1e-4
        else:
            atol, rtol = 5e-3, 5e-3
        row_layer_grads = get_grad_tensors_for_check(
            mixtral_model,
            shard_mixtral_model,
            row_layer_for_check,
            tp_group,
            atol=atol,
            rtol=rtol,
            dim=0,
            verbose=False,
        )
        col_layer_grads = get_grad_tensors_for_check(
            mixtral_model,
            shard_mixtral_model,
            col_layer_for_check,
            tp_group,
            atol=atol,
            rtol=rtol,
            dim=1,
            verbose=False,
        )
        grads_to_check.update(col_layer_grads)
        grads_to_check.update(row_layer_grads)

    # check grads
    # print(grads_to_check)
    check_all_grad_tensors(grads_to_check)
    for (n1, p1), (n2, p2) in zip(mixtral_model.named_parameters(), shard_mixtral_model.named_parameters()):
        try:
            assert_close(p1, p2, atol=5e-3, rtol=5e-3, check_dtype=False)
            print(f"{rank=},passed param before step: {n1}, {n2}")
        except Exception:
            print(
                f"{rank=},failed param before step: {n1} {p1[:2,:2] if p1 else None}, {n2} {p2[:2, :2] if p2 else None}"
            )
    # optimizer executes step
    org_optimizer.step()
    sharded_optimizer.step()
    for (n1, p1), (n2, p2) in zip(mixtral_model.named_parameters(), shard_mixtral_model.named_parameters()):
        try:
            assert_close(p1, p2, atol=5e-3, rtol=5e-3, check_dtype=False)
            print(f"{rank=},passed param after step: {n1}, {n2}")
        except Exception as e:
            print(
                f"{rank=},failed param after step: {n1} {p1 if p1 is not None else None}, {n2} {p2 if p2 is not None else None}"
            )
            raise e
    # check weights
    if stage_manager is None or stage_manager.is_first_stage():
        if test_config["precision"] == "fp32":
            atol, rtol = 2e-4, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        try:
            check_weight(
                mixtral_model,
                shard_mixtral_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
        except Exception as e:
            rank = dist.get_rank()
            print(f"{rank=}, Failed config: {test_config}")
            raise e

    torch.cuda.empty_cache()


@parameterize(
    "test_config",
    [
        # {
        #     "tp_size": 1,
        #     "pp_size": 2,
        #     "num_microbatches": 2,
        #     "ep_size": 2,
        #     "zero_stage": 1,
        #     "overlap_communication": False,
        #     "precision": "fp32",
        # },  # [dp(4)] + [moe_dp(4)]
        # {
        #     "tp_size": 1,
        #     "pp_size": 2,
        #     "num_microbatches": 2,
        #     "ep_size": 2,
        #     "zero_stage": 1,
        #     "overlap_communication": False,
        #     "precision": "fp32",
        # },  # [dp(2) + pp(2)] + [moe_pp(2)]
        # {
        #     "tp_size": 2,
        #     "pp_size": 2,
        #     "num_microbatches": 2,
        #     "ep_size": 2,
        #     "zero_stage": 1,
        #     "overlap_communication": False,
        #     "precision": "fp32",
        # },  # [pp(2) + tp(2)] + [pp(2), replicate(2)] pass
        {  # Ulysess + Flash attention
            "tp_size": 1,
            "pp_size": 1,
            "sp_size": 4,
            "ep_size": 1,
            "enable_sequence_parallelism": True,
            "sequence_parallelism_mode": "all_to_all",
            "zero_stage": 0,
            "overlap_communication": False,
            "precision": "fp16",
            "initial_scale": 1,
            "find_unused_parameters": True,
        },
        # {
        #     "tp_size": 1,
        #     "pp_size": 1,
        #     "ep_size": 2,
        #     "zero_stage": 0,
        #     "overlap_communication": False,
        #     "precision": "fp32",
        # },  # [dp(4)] + [ep(2) + moe_tp(2)]
        # {
        #     "tp_size": 1,
        #     "pp_size": 1,
        #     "ep_size": 4,
        #     "overlap_communication": False,
        #     "zero_stage": 0,
        #     "precision": "fp32"
        # },  # full dp for non-moe and full ep for moe
    ],
)
def run_mixtral_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_mixtral")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def check_mixtral(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_mixtral_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_mixtral():
    spawn(check_mixtral, 4)


if __name__ == "__main__":
    test_mixtral()
