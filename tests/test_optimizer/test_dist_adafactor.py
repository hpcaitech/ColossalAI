import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer.adafactor import Adafactor
from colossalai.nn.optimizer.distributed_adafactor import DistributedAdaFactor
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.layer.utils import Randomizer
from colossalai.tensor.d_tensor import (
    distribute_tensor,
    get_device_mesh,
    get_layout,
    get_sharding_spec,
    is_distributed_tensor,
    shard_colwise,
    shard_rowwise,
)
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.tensor.d_tensor.sharding_spec import DimSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import LowLevelZeroOptimizer
from tests.kit.model_zoo import model_zoo
from tests.test_optimizer._utils import check_dist_optim_state, check_dist_param, check_optim_states
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    build_model_from_low_level_zero_plugin,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
    run_forward_backward_with_low_level_zero_plugin,
    unwrap_model,
)

HEIGHT = 4
WIDTH = 4
_TP_SPEC = DimSpec([0])


def correctness_verify(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float32:
        rtol = 5e-04
        atol = 5e-04
    elif dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3

    assert_close(tensor1, tensor2, rtol=rtol, atol=atol)


# setup param groups; (For zero test optim)
def setup_param_groups_zero(model: nn.Module) -> list:
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


# setup param groups; (For base optim)
def setup_param_groups(model: nn.Module) -> list:
    optimizer_grouped_parameters = [p for n, p in model.named_parameters()]
    return optimizer_grouped_parameters


# setup flatten param groups, sharding spec and shape; (For dist optim)
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


def set_dist_grad(
    dist_module: nn.Module, torch_model: nn.Module, g_dtype: torch.dtype, group: dist.ProcessGroup
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
            split_dim = sharding.index(_TP_SPEC)
            shape = torch_p.split(world_size, dim=split_dim)[rank].shape

            indices = torch.arange(shape[split_dim] * rank, shape[split_dim] * (rank + 1))
            # Generate grads only for the correctly split chunk
            torch_p.grad.index_add_(split_dim, indices, torch.randn(shape, device=torch_p.device, dtype=g_dtype))

        else:
            shape = torch_p.shape
            torch_p.grad += torch.randn(shape, device=torch_p.device, dtype=g_dtype)

        # avoid inconsistent grad and param dtype error
        orig_p = p.data
        p.data = torch_p.grad.clone().to(g_dtype)
        p.grad = p.data
        p.data = orig_p


def set_master_param_to_shard_param(master_param_list) -> dict:
    master_param_to_shard_param = {id(p): p for p in master_param_list}
    return master_param_to_shard_param


class MlpModel(nn.Module):
    def __init__(self):
        super(MlpModel, self).__init__()
        self.linear1 = nn.Linear(HEIGHT, WIDTH)
        self.linear2 = nn.Linear(WIDTH, HEIGHT)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TPModel(nn.Module):
    def __init__(self, linear1, linear2, tp_group=None):
        super().__init__()
        self.linear1 = Linear1D_Col.from_native_module(
            linear1, process_group=tp_group, gather_output=False, overlap=True
        )
        self.linear2 = Linear1D_Row.from_native_module(linear2, process_group=tp_group, parallel_input=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@parameterize("dtype", [torch.float32, torch.float16, torch.bfloat16])  # torch.float32, torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(4, 1)])
def exam_dist_adafactor_base(dtype: torch.dtype, tp_zero_size: tuple[int, int]):
    tp_size, zero_size = tp_zero_size
    local_rank = dist.get_rank()
    use_zero = True if zero_size > 1 else False

    proc_mesh = ProcessGroupMesh(tp_size, zero_size)
    tp_group, dp_group = proc_mesh.get_group_along_axis(0), proc_mesh.get_group_along_axis(1)

    torch.set_default_dtype(dtype)
    set_seed(42)

    # ==============================
    # Base Case
    # ==============================
    H, W = HEIGHT, WIDTH
    model_col = nn.Linear(H, W).to(local_rank)  # Col parallel weight
    weight, bias = model_col.weight, model_col.bias

    # ==============================
    # Col Parallel
    # ==============================
    weight_col_shard = shard_colwise(weight.clone(), tp_group)
    weight_col_shard_shard_spec = get_sharding_spec(weight_col_shard)  # Shard spec
    weight_col_shard_flatten = nn.Parameter(weight_col_shard.clone().flatten().requires_grad_(True))
    bias_col_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))

    # ==============================
    # Row Parallel
    # ==============================
    weight_row_shard = shard_rowwise(weight.clone(), tp_group)
    weight_row_shard_shard_spec = get_sharding_spec(weight_row_shard)  # Shard spec
    weight_row_shard_flatten = nn.Parameter(
        weight_row_shard.clone().flatten().requires_grad_(True)
    )  # flatten input(not dtensor) to optimizer
    bias_row_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))

    # ==============================
    # Init Optimizer
    # ==============================

    # base
    optimizer_base = Adafactor([weight, bias])
    cp_dist_optim = DistributedAdaFactor([weight_col_shard_flatten, bias_col_flatten])
    rp_dist_optim = DistributedAdaFactor([weight_row_shard_flatten, bias_row_flatten])

    shard_to_param_cp = set_master_param_to_shard_param([weight_col_shard_flatten, bias_col_flatten])
    cp_dist_optim.setup_distributed(
        tp_group=tp_group,
        dp_group=dp_group,
        shard_to_working_param=shard_to_param_cp,
        use_zero=use_zero,
    )

    shard_to_param_rp = set_master_param_to_shard_param([weight_row_shard_flatten, bias_row_flatten])
    rp_dist_optim.setup_distributed(
        tp_group=tp_group,
        dp_group=dp_group,
        shard_to_working_param=shard_to_param_rp,
        use_zero=use_zero,
    )

    N_STEPS = 1
    for _ in range(N_STEPS):
        # base step
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        optimizer_base.step()

        # col parallel step
        cp_dist_optim.zero_grad()
        weight_col_shard_flatten.grad = (
            distribute_tensor(weight.grad, get_device_mesh(weight_col_shard), weight_col_shard_shard_spec)
            .clone()
            .flatten()
        )
        bias_col_flatten.grad = bias.grad.clone().flatten()
        cp_dist_optim.step()

        # row parallel step
        rp_dist_optim.zero_grad()
        weight_row_shard_flatten.grad = (
            distribute_tensor(weight.grad, get_device_mesh(weight_row_shard), weight_row_shard_shard_spec)
            .clone()
            .flatten()
        )
        bias_row_flatten.grad = bias.grad.clone().flatten()
        rp_dist_optim.step()

        weight_row_chunk = weight.t().reshape(-1, W).chunk(tp_size, dim=-1)[dist.get_rank(tp_group)].flatten()
        weight_col_chunk = weight.reshape(-1, H).chunk(tp_size, dim=-1)[dist.get_rank(tp_group)].flatten()
        # verify
        correctness_verify(weight_col_chunk, weight_col_shard_flatten, dtype)
        correctness_verify(weight_row_chunk, weight_row_shard_flatten, dtype)

    print(f"Base Test Passed")


@parameterize("dtype", [torch.float16])  # torch.float32, torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(1, 4)])  # (2, 2), (4, 1), (1, 4)
def exam_dist_adafactor_zero(dtype: torch.dtype, tp_zero_size: tuple[int, int]):
    tp_size, zero_size = tp_zero_size
    use_zero = True if zero_size > 1 else False
    local_rank = dist.get_rank()

    clear_layout_converter()

    proc_mesh = ProcessGroupMesh(tp_size, zero_size)
    tp_group, dp_group = proc_mesh.get_group_along_axis(0), proc_mesh.get_group_along_axis(1)

    torch.set_default_dtype(dtype)
    set_seed(42)

    # ==============================
    # Model Init
    # ==============================
    base_model = MlpModel().to(local_rank)
    tp_model = TPModel(copy.deepcopy(base_model.linear1), copy.deepcopy(base_model.linear2), tp_group).to(local_rank)

    base_param_group = setup_param_groups(base_model)
    tp_param_group = setup_param_groups(tp_model)
    # tp_param_group_, tp_shard_spec, tp_param_shape = setup_flatten_param_groups_sharding_spec_shape(tp_model)

    # ==============================
    # Optimizer Init
    # ==============================
    base_optim = Adafactor(base_param_group)
    dist_optim = DistributedAdaFactor(tp_param_group)

    # Setup distributed optimizer
    if zero_size > 1:
        base_optim = LowLevelZeroOptimizer(
            base_optim,
            overlap_communication=True,
            initial_scale=128,
            partition_grad=True,
            dp_process_group=dp_group,
            verbose=True,
        )

        dist_optim = LowLevelZeroOptimizer(
            dist_optim,
            overlap_communication=True,
            initial_scale=128,
            partition_grad=True,
            dp_process_group=dp_group,
            verbose=True,
        )
        shard_to_param = dist_optim.master_to_working_param  # {id(): param tensor} but flattened
        dist_optim.optim.setup_distributed(
            tp_group=tp_group,
            dp_group=dp_group,
            shard_to_working_param=shard_to_param,
            use_zero=use_zero,
        )
    else:
        shard_to_param = set_master_param_to_shard_param(tp_param_group)
        dist_optim.setup_distributed(
            tp_group=tp_group,
            dp_group=dp_group,
            shard_to_working_param=shard_to_param,
            use_zero=use_zero,
        )

    # ==============================
    # Correctness Verify
    # ==============================
    x = torch.randn(HEIGHT, WIDTH, device=local_rank)

    out = base_model(x)
    out_tp = tp_model(x)

    if zero_size > 1:
        dist_optim.backward(out_tp.sum())
        base_optim.backward(out.sum())
    else:
        out_tp.sum().backward()
        out.sum().backward()

    base_optim.step()
    dist_optim.step()

    base_optim.zero_grad()
    dist_optim.zero_grad()

    for p, tp_p in zip(base_param_group, tp_param_group):
        param_is_distributed = is_distributed_tensor(tp_p)
        if param_is_distributed:
            shard_spec = get_sharding_spec(tp_p)
            if len(shard_spec.sharding_sequence) >= 2:
                # Col Parallel
                if shard_spec.sharding_sequence[0] == "R":
                    p = p.chunk(tp_size, dim=-1)[dist.get_rank(tp_group)]
                # ROW Parallel
                if shard_spec.sharding_sequence[-1] == "R":
                    p = p.chunk(tp_size, dim=0)[dist.get_rank(tp_group)]
            else:
                # TP bias
                p = p.chunk(tp_size, dim=-1)[dist.get_rank(tp_group)]

        correctness_verify(p, tp_p, dtype)
    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()
    print(f"Zero Test Passed")


@parameterize(
    "test_config",
    [
        {
            "stage": 1,
            "precision": "bf16",
        },
        {
            "stage": 2,
            "precision": "bf16",
        },
    ],
)
def exam_bert_test_on_lowlevelzero_plugin(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    model_list = [
        "transformers_bert",
    ]
    clear_layout_converter()
    torch.set_default_dtype(torch.bfloat16)
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name in model_list:
            (
                org_model,
                org_optimizer,
                sharded_model,
                sharded_optimizer,
                criterion,
                booster,
            ) = build_model_from_low_level_zero_plugin(model_fn, loss_fn, test_config, Adafactor, Adafactor)

            org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_low_level_zero_plugin(
                org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
            )

            # LowLevelZero not need warp
            # bert = unwrap_model(org_model, "BertModel", "bert")
            # sharded_bert = unwrap_model(sharded_model, "BertModel", "bert")
            weight_layer_for_check = [
                "bert.encoder.layer.0.output.dense.weight",
                "bert.encoder.layer.0.output.dense.weight",
            ]

            org_optimizer.step()
            sharded_optimizer.step()

            # check weights
            if test_config["precision"] == "bf16":
                atol, rtol = 5e-4, 5e-4
            else:
                atol, rtol = 5e-4, 5e-4

            check_dist_param(org_model, sharded_model, weight_layer_for_check, atol, rtol)
            check_optim_states(org_optimizer, sharded_optimizer.optim)

    Randomizer.reset_index()
    torch.cuda.empty_cache()
    print(f"Bert Model Zoo Test Passed")


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
            "tp_size": 2,
            "num_microbatches": 4,
            "zero_stage": 1,
            "precision": "bf16",
        },
        # @duanjunwen TODO: fix this test case. Currently params are sharded but are not dtensor here, throwing an error.
        # Probably due to HybridParallelAMPOptimizer replacing some master params ?
        # {
        #     "tp_size": 4,
        #     "num_microbatches": 4,
        #     "zero_stage": 0,
        #     "precision": "bf16",
        # },
    ],
)
def exam_bert_test_on_hybrid_plugin(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["pp_size"] = 1  # Do NOT test Pipeline Parallel
    test_config["initial_scale"] = 2**16  # avoid overflow
    model_list = [
        "transformers_bert",
    ]
    clear_layout_converter()
    torch.set_default_dtype(torch.bfloat16)
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name in model_list:
            (
                org_model,
                org_optimizer,
                sharded_model,
                sharded_optimizer,
                criterion,
                booster,
            ) = build_model_from_hybrid_plugin(model_fn, loss_fn, test_config, Adafactor, DistributedAdaFactor)

            org_loss, org_output, sharded_loss, sharded_output = run_forward_backward_with_hybrid_plugin(
                org_model, sharded_model, sharded_optimizer, data_gen_fn, output_transform_fn, criterion, booster
            )

            stage_manager = booster.plugin.stage_manager
            tp_group = booster.plugin.tp_group

            bert = unwrap_model(org_model, "BertModel", "bert")
            sharded_bert = unwrap_model(sharded_model, "BertModel", "bert")
            weight_layer_for_check = ["encoder.layer[0].output.dense", "encoder.layer[1].output.dense"]

            org_optimizer.step()
            sharded_optimizer.step()

            # check weights
            if test_config["precision"] == "bf16":
                atol, rtol = 5e-4, 5e-4
            else:
                atol, rtol = 5e-4, 5e-4
            if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
                check_weight(bert, sharded_bert, weight_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1)
                # check optim states
                check_dist_optim_state(org_optimizer, sharded_optimizer.optim)

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()
    print(f"Bert Model Zoo Test Passed")


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    exam_dist_adafactor_base()
    exam_dist_adafactor_zero()
    exam_bert_test_on_lowlevelzero_plugin()
    exam_bert_test_on_hybrid_plugin()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_adafactor():
    spawn(run_dist, nprocs=4)


if __name__ == "__main__":
    test_dist_adafactor()
