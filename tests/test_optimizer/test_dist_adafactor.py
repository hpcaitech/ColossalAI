import copy
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.testing import assert_close

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin, HybridParallelPlugin
from colossalai.cluster import ProcessGroupMesh
from colossalai.device.device_mesh import DeviceMesh
from colossalai.nn.optimizer.adafactor import Adafactor
from colossalai.nn.optimizer.distributed_adafactor import DistributedAdaFactor
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.shardformer.layer._operation import _gather
from colossalai.tensor.d_tensor import (
    distribute_tensor,
    get_layout,
    get_sharding_spec,
    get_device_mesh,
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
from tests.test_optimizer._utils import run_bert_test
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
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

    # return torch.all(tensor1.isclose(tensor2, rtol=rtol, atol=atol))
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
    # print(f"sharding_spec {sharding_spec}")
    # print(f"param_shape {param_shape}")
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
    master_param_to_shard_param ={id(p):p for p in master_param_list}
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




@parameterize("dtype", [torch.float32])  # , torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(4, 1)])  # (2, 2), (4, 1),(1, 4), (2, 4), (4, 2)
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
    H, W = 4096, 4096
    model_col = nn.Linear(H, W).to(local_rank)  # Col parallel weight
    weight, bias = model_col.weight, model_col.bias
    
    # ==============================
    # Col Parallel
    # ==============================
    weight_col_shard = shard_colwise(weight.clone(), tp_group)
    weight_col_shard_layout = get_layout(weight_col_shard)  # Layout info weight_col_shard_layout.global_shape
    weight_col_shard_shard_spec = get_sharding_spec(weight_col_shard)  # Shard spec
    weight_col_shard_flatten = nn.Parameter(weight_col_shard.clone().flatten().requires_grad_(True))
    bias_col_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))

    # ==============================
    # Row Parallel
    # ==============================
    weight_row_shard = shard_rowwise(weight.clone(), tp_group)
    weight_row_shard_layout = get_layout(weight_row_shard)  # Layout info weight_row_shard_layout.global_shape
    weight_row_shard_shard_spec = get_sharding_spec(weight_row_shard)  # Shard spec
    weight_row_shard_flatten = nn.Parameter(
        weight_row_shard.clone().flatten().requires_grad_(True)
    )  # flatten input(not dtensor) to optimizer
    bias_row_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
    
    base_param_group = setup_param_groups([weight, bias])
    cp_param_group = setup_param_groups([weight_col_shard_flatten, bias_col_flatten])
    rp_param_group = setup_param_groups([weight_row_shard_flatten, bias_row_flatten])

    # ==============================
    # Init Optimizer
    # ==============================

    # base
    optimizer_base = Adafactor(base_param_group)
    cp_dist_optim = DistributedAdaFactor(cp_param_group)
    rp_dist_optim = DistributedAdaFactor(rp_param_group)
    
    shard_to_param_cp = set_master_param_to_shard_param(cp_dist_optim)
    cp_dist_optim.setup_distributed(
        tensor_parallel_group=tp_group,
        data_parallel_group=dp_group,
        shard_to_param=shard_to_param_cp,
        use_zero=use_zero,
    )
    
    shard_to_param_rp = set_master_param_to_shard_param(rp_dist_optim)
    rp_dist_optim.setup_distributed(
        tensor_parallel_group=tp_group,
        data_parallel_group=dp_group,
        shard_to_param=shard_to_param_rp,
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
            distribute_tensor(weight.grad, get_device_mesh(weight_col_shard), weight_col_shard_shard_spec).clone().flatten()
        )
        bias_col_flatten.grad = bias.grad.clone().flatten()
        cp_dist_optim.step()

        # row parallel step
        rp_dist_optim.zero_grad()
        weight_row_shard_flatten.grad = (
            distribute_tensor(weight.grad, get_device_mesh(weight_row_shard), weight_row_shard_shard_spec).clone().flatten()
        )
        bias_row_flatten.grad = bias.grad.clone().flatten()
        rp_dist_optim.step()

        # gather result
        weight_col_gather = _gather(
            input_=weight_col_shard_flatten.data.view(-1, H // tp_size),
            dim=-1,
            process_group=tp_group,
        )  # gather
        weight_row_gather = _gather(
            input_=weight_row_shard_flatten.data, dim=-1, process_group=tp_group
        ).view(
            -1, W
        )  # gather

        # verify
        col_correct = correctness_verify(weight.data, weight_col_gather.data, dtype)
        row_correct = correctness_verify(weight.data, weight_row_gather.data, dtype)

        print(f"col corrness {col_correct}  row correct {row_correct}")


@parameterize("dtype", [torch.float32])  # , torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(4, 1)])  # (2, 2), (4, 1),(1, 4), (2, 4), (4, 2)
def exam_dist_adafactor_fwd_bwd(dtype: torch.dtype, tp_zero_size: tuple[int, int]):
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

    # ==============================
    # Optimizer Init
    # ==============================
    base_optim = Adafactor(base_param_group)
    dist_optim = DistributedAdaFactor(tp_param_group)
    
    shard_to_param = set_master_param_to_shard_param(tp_param_group)
    dist_optim.setup_distributed(
        tensor_parallel_group=tp_group,
        data_parallel_group=dp_group,
        shard_to_param=shard_to_param,
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
            # print(f"device {local_rank} shard spec{shard_spec} len {len(shard_spec.sharding_sequence)}\n")
            if len(shard_spec.sharding_sequence) >= 2:
                # Col Parallel
                if shard_spec.sharding_sequence[0] == "R":
                    tp_p = _gather(input_=tp_p, dim=-1, process_group=tp_group)  # gather
                # ROW Parallel
                if shard_spec.sharding_sequence[-1] == "R":
                    tp_p = _gather(input_=tp_p, dim=0, process_group=tp_group)  # gather
            else:
                # TP bias
                tp_p = _gather(input_=tp_p, dim=-1, process_group=tp_group)  # gather

        else:
            # No TP bias
            pass
        correctness = correctness_verify(p.data, tp_p.data, dtype)
        # print(f"{correctness}\n p.data {p.data}\n tp_p.data{tp_p.data}\n")
        # print(f"Curr Param correct {correctness}")
    # print(f"device {local_rank} base_optim state dict {base_optim.optim.state_dict()['state'].items()} \n dist_optim state dict {dist_optim.optim.state_dict()['state'].items()} \n")


@parameterize("dtype", [torch.float32, torch.float16, torch.bfloat16])  # torch.float32, torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(1, 4), (4, 1), (2, 2)])  # (2, 2), (4, 1),(1, 4), (2, 4), (4, 2)
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
    tp_param_group_, tp_shard_spec, tp_param_shape = setup_flatten_param_groups_sharding_spec_shape(tp_model)

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
        shard_to_param = dist_optim._param_store.master_to_working_param  # {id(): param tensor} but flattened
        dist_optim.optim.setup_distributed(
            tensor_parallel_group=tp_group,
            data_parallel_group=dp_group,
            shard_to_param=shard_to_param,
            use_zero=use_zero,
        )
    else:
        shard_to_param = set_master_param_to_shard_param(tp_param_group)
        dist_optim.setup_distributed(
            tensor_parallel_group=tp_group,
            data_parallel_group=dp_group,
            shard_to_param=shard_to_param,
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

    print(f"data type {dtype},tp size {tp_size}, dp size {zero_size}\n")
    for p, tp_p in zip(base_param_group, tp_param_group):
        param_is_distributed = is_distributed_tensor(tp_p)
        if param_is_distributed:
            shard_spec = get_sharding_spec(tp_p)
            # print(f"device {local_rank} shard spec{shard_spec} len {len(shard_spec.sharding_sequence)}\n")
            if len(shard_spec.sharding_sequence) >= 2:
                # Col Parallel
                if shard_spec.sharding_sequence[0] == "R":
                    tp_p = _gather(input_=tp_p, dim=-1, process_group=tp_group)  # gather
                # ROW Parallel
                if shard_spec.sharding_sequence[-1] == "R":
                    tp_p = _gather(input_=tp_p, dim=0, process_group=tp_group)  # gather
            else:
                # TP bias
                tp_p = _gather(input_=tp_p, dim=-1, process_group=tp_group)  # gather

        else:
            # No TP bias
            pass
        correctness = correctness_verify(p.data, tp_p.data, dtype)
        
        # print(f"Curr Param correct {correctness}")
        # if not correctness:
        #     print(f"{correctness}\n p.data {p.data}\n tp_p.data{tp_p.data}\n")


@parameterize("dtype", [torch.float32, torch.float16, torch.bfloat16])  # torch.float32, torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(1, 4), (4, 1), (2, 2)])  # (2, 2), (4, 1),(1, 4), (2, 4), (4, 2)
def exam_dist_adafactor_booster(dtype: torch.dtype, tp_zero_size: tuple[int, int]):
    tp_size, zero_size = tp_zero_size
    local_rank = dist.get_rank()
    use_zero = True if zero_size > 1 else False
    
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
        shard_to_param = dist_optim._param_store.master_to_working_param  # {id(): param tensor} but flattened
        dist_optim.optim.setup_distributed(
            tensor_parallel_group=tp_group,
            data_parallel_group=dp_group,
            shard_to_param=shard_to_param,
            use_zero=use_zero,
        )
    else:
        shard_to_param = set_master_param_to_shard_param(tp_param_group)
        dist_optim.setup_distributed(
            tensor_parallel_group=tp_group,
            data_parallel_group=dp_group,
            shard_to_param=shard_to_param,
            use_zero=use_zero,
        )

    # ==============================
    # Booster Init
    # ==============================
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    criterion = lambda x: x.mean()

    tp_model, dist_optim, criterion, _, _ = booster.boost(tp_model, dist_optim, criterion)

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
    print(f"data type {dtype},tp size {tp_size}, dp size {zero_size}\n")
    for p, tp_p in zip(base_param_group, tp_param_group):
        param_is_distributed = is_distributed_tensor(tp_p)
        if param_is_distributed:
            shard_spec = get_sharding_spec(tp_p)
            # print(f"device {local_rank} shard spec{shard_spec} len {len(shard_spec.sharding_sequence)}\n")
            if len(shard_spec.sharding_sequence) >= 2:
                # Col Parallel
                if shard_spec.sharding_sequence[0] == "R":
                    tp_p = _gather(input_=tp_p, dim=-1, process_group=tp_group)  # gather
                # ROW Parallel
                if shard_spec.sharding_sequence[-1] == "R":
                    tp_p = _gather(input_=tp_p, dim=0, process_group=tp_group)  # gather
            else:
                # TP bias
                tp_p = _gather(input_=tp_p, dim=-1, process_group=tp_group)  # gather

        else:
            # No TP bias
            pass
        correctness = correctness_verify(p.data, tp_p.data, dtype)
 
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
        }
    ],
)
def exam_bert_test(test_config):
    sub_model_zoo = model_zoo.get_sub_registry("transformers_bert")
    test_config["use_lazy_init"] = False
    test_config["pp_size"] = 1  # Do NOT test Pipeline Parallel
    test_config["initial_scale"] = 2**15  # avoid overflow

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        
        if name == "transformers_bert":
            org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
                model_fn, loss_fn, test_config, Adafactor, DistributedAdaFactor
            )
            
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
                atol, rtol = 5e-4, 1e-4
            else:
                atol, rtol = 5e-4, 5e-4
            if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
                check_weight(bert, sharded_bert, weight_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1)
        clear_layout_converter()
        torch.cuda.empty_cache()



def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # exam_dist_adafactor_base()
    # exam_dist_adafactor_fwd_bwd()
    exam_dist_adafactor_zero()
    exam_bert_test()



@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_adafactor():
    spawn(run_dist, nprocs=4)


if __name__ == "__main__":
    test_dist_adafactor()
