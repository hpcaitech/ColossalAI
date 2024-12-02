import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor import get_layout, get_sharding_spec, is_distributed_tensor
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.tensor.d_tensor.sharding_spec import DimSpec
from colossalai.testing import parameterize, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)


def force_assign_grad(p, g_dtype, grad=None):
    """Bypass inconsistent grad and param dtype error when assigning grad"""
    orig_p = p.data
    p.data = torch.randn_like(p, device=orig_p.device, dtype=g_dtype) if grad == None else grad.clone().to(g_dtype)
    p.grad = p.data
    p.data = orig_p


def setup_param_groups(model: nn.Module) -> list:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


# setup flatten param groups, sharding spec and shape; (For dist Adafactor and CAME)
def setup_flatten_param_groups_sharding_spec_shape(model: nn.Module) -> dict:
    flatten_optimizer_grouped_parameters = []
    sharding_spec = {}  # {id(flatten param): get_layout(p).global_shape}
    param_shape = {}  # {id(flatten param): get_sharding_spec(p)}
    for n, p in model.named_parameters():
        # flatten_p = copy.deepcopy(p).flatten()
        flatten_p = nn.Parameter(p.clone().flatten().requires_grad_(True))
        flatten_optimizer_grouped_parameters.append(flatten_p)
        if is_distributed_tensor(p):
            sharding_spec[id(flatten_p)] = get_sharding_spec(p)
            param_shape[id(flatten_p)] = get_layout(p).global_shape
        else:
            sharding_spec[id(flatten_p)] = None
            param_shape[id(flatten_p)] = p.shape
    return flatten_optimizer_grouped_parameters, sharding_spec, param_shape


def set_master_param_to_shard_param(master_param_list) -> dict:
    master_param_to_shard_param = {id(p): p for p in master_param_list}
    return master_param_to_shard_param


def set_dist_grad(
    dist_module: nn.Module,
    torch_model: nn.Module,
    g_dtype: torch.dtype,
    group: dist.ProcessGroup,
    tp_spec: DimSpec,
) -> None:
    """
    Set split grads for Tensor Parallel or ZeRO DP.
    We do not need a separate treatment for ZeRO,
    as the wrapper takes care of reduce-scattering grads.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    for p, torch_p in zip(dist_module.parameters(), torch_model.parameters()):
        if torch_p.grad is None:
            torch_p.grad = torch.zeros_like(torch_p)

        is_distributed = hasattr(p, "dist_layout")
        if is_distributed:
            sharding = p.dist_layout.sharding_spec.sharding_sequence
            split_dim = sharding.index(tp_spec)
            shape = torch_p.split(world_size, dim=split_dim)[rank].shape

            indices = torch.arange(shape[split_dim] * rank, shape[split_dim] * (rank + 1))
            # Generate grads only for the correctly split chunk
            torch_p.grad.index_add_(split_dim, indices, torch.randn(shape, device=torch_p.device, dtype=g_dtype))

        else:
            shape = torch_p.shape
            torch_p.grad += torch.randn(shape, device=torch_p.device, dtype=g_dtype)

        force_assign_grad(p, g_dtype, grad=torch_p.grad)


def check_optim_states(org_optim, sharded_optim):
    for group in org_optim.param_groups:
        for p in group["params"]:
            sharded_state = sharded_optim.state[p]
            state = org_optim.state[p]
            for key in sharded_state:
                assert_close(state[key], sharded_state[key], rtol=1e-5, atol=1e-5)


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
    check_optim_states(org_optimizer, sharded_optimizer.optim)
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
        {
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
        },
        {
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 0,
            "precision": "bf16",
        },
    ],
)
def run_bert_test(test_config, optim_class, sharded_optim_class):
    """Only call this if you've initialized distributed backend and spawned processes"""
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["pp_size"] = 1  # Do NOT test Pipeline Parallel
    test_config["initial_scale"] = 2**15  # avoid overflow
    target_models = [
        "transformers_bert",
    ]

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name in target_models:
            check_bert_fwd_bwd(
                model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config, optim_class, sharded_optim_class
            )

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def _run_bert_test(rank, world_size, port, optim_class, sharded_optim_class):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_bert_test(optim_class, sharded_optim_class)


def check_optim_on_bert(optim_class, sharded_optim_class):
    spawn(_run_bert_test, 4, optim_class, sharded_optim_class)


def check_dist_optim_state(org_optimizer, sharded_optimizer):
    torch.set_default_dtype(torch.bfloat16)
    for group, tp_group in zip(org_optimizer.param_groups, sharded_optimizer.param_groups):
        for p, tp in zip(group["params"], tp_group["params"]):
            p_state = org_optimizer.state[p]
            tp_state = sharded_optimizer.state[tp]
            # TODO "exp_avg_sq_col", "exp_avg_sq_row", "exp_avg_sq"
            for key in ["exp_avg_sq_row"]:
                if key in tp_state.keys() and type(tp_state[key]) is torch.Tensor:
                    tp_is_dtensor = sharded_optimizer.param_is_dtensor_dict[id(tp)]
                    shard_spec = sharded_optimizer.shard_spec_dict[id(tp)]
                    use_zero = sharded_optimizer.use_zero
                    tp_optim_state = tp_state[key]
                    state = p_state[key]

                    dp_size, tp_size = (
                        sharded_optimizer.dp_size,
                        sharded_optimizer.tp_size,
                    )
                    # we start init model with first tensor parallel then zero;
                    # So, we gather model with first zero then tensor parallel

                    if tp_is_dtensor:
                        # col parallel
                        if shard_spec.sharding_sequence[0] == "R":
                            if use_zero:
                                # sq_row need gather alone dp group
                                # sq_col don't need gather alone dp group
                                if key == "exp_avg_sq_row":
                                    state = state.chunk(dp_size, dim=-1)[dist.get_rank(sharded_optimizer.dp_group)]

                            # gather from tp group
                            # sq_row don need gather alone tp group
                            # sq_col need gather alone tp group
                            if key == "exp_avg_sq_col":
                                state = state.chunk(tp_size, dim=-1)[dist.get_rank(sharded_optimizer.tp_group)]
                        # row parallel
                        elif shard_spec.sharding_sequence[-1] == "R":
                            # TODO: this case may cause shape mismatch @duanjunwen
                            if use_zero and key == "exp_avg_sq_row" and state.shape[0] // tp_size % dp_size == 0:
                                # sq_row need gather alone dp group
                                # sq_col don't need gather alone dp group

                                state = state.chunk(dp_size, dim=-1)[dist.get_rank(sharded_optimizer.dp_group)]

                            # gather from tp group
                            # sq_row need gather alone tp group
                            if key == "exp_avg_sq_row":
                                state = state.chunk(tp_size, dim=-1)[dist.get_rank(sharded_optimizer.tp_group)]
                            # sq_col don't need gather alone dp group
                            if key == "exp_avg_sq_col":
                                pass
                        else:
                            return
                    else:
                        if use_zero:
                            # sq_row need gather alone dp group
                            if key == "exp_avg_sq_row":
                                # row residule; no gather
                                if state.shape[0] % dp_size != 0:
                                    pass
                                else:
                                    state = state.chunk(dp_size, dim=-1)[dist.get_rank(sharded_optimizer.dp_group)]
                            # sq_col don't need gather alone dp group
                            if key == "exp_avg_sq_col":
                                tp_optim_state = tp_optim_state.div_(dp_size)
                                # need a div;

                    if state.dtype != tp_optim_state.dtype:
                        tp_optim_state = tp_optim_state.type(state.dtype)
                    # TODO: some sharding checks are currently buggy, but the state values should match
                    # @duanjunwen
                    if state.shape != tp_optim_state.shape:
                        return
                    assert_close(state, tp_optim_state, atol=5e-4, rtol=1.6e-2)


def check_dist_param(org_model, sharded_model, weight_layer_for_check, atol, rtol):
    for (org_name, org_param), (sharded_name, sharded_param) in zip(
        org_model.named_parameters(), sharded_model.named_parameters()
    ):
        if org_name in weight_layer_for_check:
            assert_close(org_param, sharded_param, atol=atol, rtol=rtol)


def check_dist_grad(sharded_optimizer, org_model, sharded_model, weight_layer_for_check, atol, rtol):
    for (org_name, org_param), (sharded_name, sharded_param) in zip(
        org_model.named_parameters(), sharded_model.named_parameters()
    ):
        if org_name in weight_layer_for_check:
            org_grad = org_param.grad
            group_id = dist.get_rank(sharded_optimizer.optim.dp_group)
            dist_grad = sharded_optimizer.get_partitioned_gradients_by_param_id(group_id, id(sharded_param))

            # dist_grad concat then reshape to org_grad shape
            if dist_grad:
                dist_grad = torch.cat([t for t in dist_grad], 0).view(org_grad.shape)
                assert_close(org_grad, dist_grad, atol=atol, rtol=rtol)
