import copy
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
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
    is_distributed_tensor,
    shard_colwise,
    shard_rowwise,
)
from colossalai.tensor.d_tensor.sharding_spec import DimSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from colossalai.zero import LowLevelZeroOptimizer

HEIGHT = 4096
WIDTH = 4096
_TP_SPEC = DimSpec([0])


def correctness_verify(tensor1: torch.Tensor, tensor2: torch.Tensor, dtype: torch.dtype = torch.float32):
    rtol = None
    atol = None
    if dtype is torch.float32:
        rtol = 1e-05
        atol = 1e-05
    elif dtype is torch.float16:
        rtol = 5e-2
        atol = 5e-4
    elif dtype is torch.bfloat16:
        rtol = 4e-3
        atol = 4e-3

    return torch.all(tensor1.isclose(tensor2, rtol=rtol, atol=atol))
    # assert_close(tensor1, tensor2, rtol=rtol, atol=atol)


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
def exam_dist_adafactor_base(dtype: torch.dtype):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    tensor_parallel_size = world_size
    torch.set_default_dtype(dtype)
    set_seed(42)

    # ==============================
    # Base Case
    # ==============================
    H, W = 4096, 4096
    model_col = nn.Linear(H, W).to(local_rank)  # Col parallel weight
    weight, bias = model_col.weight, model_col.bias
    device_mesh = DeviceMesh(
        torch.Tensor([i for i in range(world_size)]), (1, tensor_parallel_size), init_process_group=True
    )
    tp_group = device_mesh.get_process_group(axis=1)
    # ==============================
    # Col Parallel
    # ==============================
    weight_col_shard = shard_colwise(weight.clone(), device_mesh.get_process_group(axis=1))
    weight_col_shard_layout = get_layout(weight_col_shard)  # Layout info weight_col_shard_layout.global_shape
    weight_col_shard_shard_spec = get_sharding_spec(weight_col_shard)  # Shard spec
    weight_col_shard_flatten = nn.Parameter(weight_col_shard.clone().flatten().requires_grad_(True))
    bias_col_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
    col_params_shape = {
        id(weight_col_shard_flatten): weight_col_shard_layout.global_shape,
        id(bias_col_flatten): bias.shape,
    }
    col_sharding_spec_dict = {id(weight_col_shard_flatten): weight_col_shard_shard_spec, id(bias_col_flatten): None}

    # ==============================
    # Row Parallel
    # ==============================
    weight_row_shard = shard_rowwise(weight.clone(), device_mesh.get_process_group(axis=1))
    weight_row_shard_layout = get_layout(weight_row_shard)  # Layout info weight_row_shard_layout.global_shape
    weight_row_shard_shard_spec = get_sharding_spec(weight_row_shard)  # Shard spec
    weight_row_shard_flatten = nn.Parameter(
        weight_row_shard.clone().flatten().requires_grad_(True)
    )  # flatten input(not dtensor) to optimizer
    bias_row_flatten = nn.Parameter(bias.clone().flatten().requires_grad_(True))
    row_params_shape = {
        id(weight_row_shard_flatten): weight_row_shard_layout.global_shape,
        id(bias_row_flatten): bias.shape,
    }
    row_sharding_spec_dict = {id(weight_row_shard_flatten): weight_row_shard_shard_spec, id(bias_row_flatten): None}

    # ==============================
    # Init Optimizer
    # ==============================

    # base
    optimizer_base = Adafactor([weight, bias])

    # col parallel
    optimizer_cp = DistributedAdaFactor([weight_col_shard_flatten, bias_col_flatten])
    optimizer_cp.setup_distributed(
        tensor_parallel_group=tp_group,
        data_parallel_group=None,
        sharding_spec_dict=col_sharding_spec_dict,
        param_shape=col_params_shape,
    )
    # row parallel
    optimizer_rp = DistributedAdaFactor([weight_row_shard_flatten, bias_row_flatten])
    optimizer_rp.setup_distributed(
        tensor_parallel_group=tp_group,
        data_parallel_group=None,
        sharding_spec_dict=row_sharding_spec_dict,
        param_shape=row_params_shape,
    )

    N_STEPS = 1
    for _ in range(N_STEPS):
        # base step
        optimizer_base.zero_grad()
        weight.grad = torch.rand_like(weight)
        bias.grad = torch.rand_like(bias)
        optimizer_base.step()

        # col parallel step
        optimizer_cp.zero_grad()
        weight_col_shard_flatten.grad = (
            distribute_tensor(weight.grad, device_mesh, weight_col_shard_shard_spec).clone().flatten()
        )
        bias_col_flatten.grad = bias.grad.clone().flatten()
        optimizer_cp.step()

        # row parallel step
        optimizer_rp.zero_grad()
        weight_row_shard_flatten.grad = (
            distribute_tensor(weight.grad, device_mesh, weight_row_shard_shard_spec).clone().flatten()
        )
        bias_row_flatten.grad = bias.grad.clone().flatten()
        optimizer_rp.step()

        # gather result
        weight_col_gather = _gather(
            input_=weight_col_shard_flatten.data.view(-1, H // tensor_parallel_size),
            dim=-1,
            process_group=device_mesh.get_process_group(axis=1),
        )  # gather
        weight_row_gather = _gather(
            input_=weight_row_shard_flatten.data, dim=-1, process_group=device_mesh.get_process_group(axis=1)
        ).view(
            -1, W
        )  # gather

        # verify
        col_correct = correctness_verify(weight.data, weight_col_gather.data, dtype)
        row_correct = correctness_verify(weight.data, weight_row_gather.data, dtype)

        print(f"col corrness {col_correct}  row correct {row_correct}")


@parameterize("dtype", [torch.float32])  # , torch.float16, torch.bfloat16
def exam_dist_adafactor_fwd_bwd(dtype: torch.dtype):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    tensor_parallel_size = world_size
    torch.set_default_dtype(dtype)
    set_seed(42)

    # ==============================
    # Model Init
    # ==============================
    device_mesh = DeviceMesh(
        torch.Tensor([i for i in range(world_size)]), (1, tensor_parallel_size), init_process_group=True
    )
    base_model = MlpModel().to(local_rank)
    tp_model = TPModel(
        copy.deepcopy(base_model.linear1), copy.deepcopy(base_model.linear2), device_mesh.get_process_group(axis=1)
    ).to(local_rank)
    tp_group = device_mesh.get_process_group(axis=1)

    base_param_group = setup_param_groups(base_model)
    tp_param_group, tp_shard_spec, tp_param_shape = setup_flatten_param_groups_sharding_spec_shape(tp_model)

    # ==============================
    # Optimizer Init
    # ==============================
    base_optim = Adafactor(base_param_group)
    dist_optim = DistributedAdaFactor(tp_param_group)
    dist_optim.setup_distributed(
        tensor_parallel_group=tp_group,
        data_parallel_group=None,
        sharding_spec_dict=tp_shard_spec,
        param_shape=tp_param_shape,
    )

    # ==============================
    # Correctness Verify
    # ==============================
    x = torch.randn(HEIGHT, WIDTH, device=local_rank)

    loss_tp = tp_model(x).sum()
    loss_tp.backward()

    loss = base_model(x).sum()
    loss.backward()

    base_optim.zero_grad()
    dist_optim.zero_grad()

    base_optim.step()
    dist_optim.step()

    for p, tp_p in zip(base_param_group, tp_param_group):
        if tp_shard_spec[id(tp_p)] is not None:
            if len(tp_shard_spec[id(tp_p)].sharding_sequence) >= 2:
                # print(f"device {local_rank} \n  tp_p shard spec {tp_shard_spec[id(tp_p)]}\n len {len(tp_shard_spec[id(tp_p)].sharding_sequence)}")
                # if tp_p tp_shard_spec is col tp --> view to (-1, H // tensor_parallel_size) then gather
                if tp_shard_spec[id(tp_p)].sharding_sequence[0] == "R":
                    tp_p = _gather(
                        input_=tp_p.data.view(-1, HEIGHT // tensor_parallel_size),
                        dim=-1,
                        process_group=device_mesh.get_process_group(axis=1),
                    )  # gather
                # if tp_p tp_shard_spec is row tp  --> gather then view to (-1, H // tensor_parallel_size)
                else:
                    tp_p = _gather(input_=tp_p.data, dim=-1, process_group=device_mesh.get_process_group(axis=1)).view(
                        -1, WIDTH
                    )  # gather
            else:
                # bias parallel
                tp_p = _gather(input_=tp_p.data, dim=-1, process_group=device_mesh.get_process_group(axis=1))
                # print(f"device {local_rank} \n p {p}\n tp_p {tp_p}\n")
        else:
            # compare p and tp no need
            pass
        # print(f"device {local_rank} \n p {p}\n tp_p {tp_p}\n")
        correctness_verify(p.data, tp_p.data, dtype)
        # print(f"correct {correctness}")


@parameterize("dtype", [torch.bfloat16])  # torch.float32, torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(4, 2)])  # (2, 2), (4, 1),(1, 4), (2, 4), (4, 2)
def exam_dist_adafactor_zero(dtype: torch.dtype, tp_zero_size: tuple[int, int]):
    tp_size, zero_size = tp_zero_size
    use_zero = True if zero_size > 1 else False
    local_rank = dist.get_rank()

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
        print(f"Curr Param correct {correctness}")
    # print(f"device {local_rank} base_optim state dict {base_optim.optim.state_dict()['state'].items()} \n dist_optim state dict {dist_optim.optim.state_dict()['state'].items()} \n")

    
    


@parameterize("dtype", [torch.bfloat16])  # torch.float32, torch.float16, torch.bfloat16
@parameterize("tp_zero_size", [(4, 2)])  # (2, 2), (4, 1),(1, 4), (2, 4), (4, 2)
def exam_dist_adafactor_booster(dtype: torch.dtype, tp_zero_size: tuple[int, int]):
    tp_size, zero_size = tp_zero_size
    local_rank = dist.get_rank()
    use_zero = True if zero_size > 1 else False

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
        print(f"Curr Param correct {correctness}")


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    # exam_dist_adafactor_base()
    # exam_dist_adafactor_fwd_bwd()
    exam_dist_adafactor_zero()
    # exam_dist_adafactor_booster()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_adafactor():
    spawn(run_dist, nprocs=8)


if __name__ == "__main__":
    test_dist_adafactor()
