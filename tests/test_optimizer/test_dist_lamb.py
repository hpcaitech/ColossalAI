from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

import colossalai
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import DistributedLamb, Lamb
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.tensor.d_tensor import is_distributed_tensor
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.tensor.d_tensor.sharding_spec import DimSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero import LowLevelZeroOptimizer
from tests.test_optimizer._utils import run_bert_test

_ALLOWED_P_G_TYPES = [
    (torch.float, torch.float),  # pure fp32
    (torch.float, torch.half),  # fp16 amp
    (torch.float, torch.bfloat16),  # bfloat16 amp
]

# Identifiers for Tensor Parallel linear layers
_SHARD_DIM = DimSpec([0])
_BS = 16
_N_STEP = 3
_IN_DIM = 32
_HID_DIM = 128
_SEED = 1024
_COORD = None


def get_split_dim(p):
    if not is_distributed_tensor(p):
        raise ValueError("p is not a distributed tensor")
    sharding = p.dist_layout.sharding_spec.sharding_sequence
    return sharding.index(_SHARD_DIM)


def assert_distributed_close(tp_model, torch_model, rtol, atol, tp_group):
    rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)

    for (name, p), torch_p in zip(tp_model.named_parameters(), torch_model.parameters()):
        # if overflow, the weight won't be updated. so there will be no nan in p
        assert not torch.isnan(p).any()
        try:
            if is_distributed_tensor(p):
                split_dim = get_split_dim(p)
                torch_p = torch_p.chunk(tp_size, dim=split_dim)[rank]

            assert_close(p.float(), torch_p, rtol=rtol, atol=atol)
        except AssertionError as e:
            print(f"grad mismatch in {name}")
            raise e


def setup_param_groups(bert_model: nn.Module) -> list:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


class Net(nn.Module):
    def __init__(self, identity=False):
        super().__init__()
        if identity:
            self.fc0 = nn.Identity()
        else:
            self.fc0 = nn.Linear(_IN_DIM, _IN_DIM)

        self.fc1 = nn.Linear(_IN_DIM, _HID_DIM)
        self.fc2 = nn.Linear(_HID_DIM, _IN_DIM)

    def forward(self, x):
        return self.fc2(self.fc1(self.fc0(x)))


class TPNet(nn.Module):
    def __init__(self, fc0, fc1, fc2, tp_group=None):
        super().__init__()
        self.fc0 = deepcopy(fc0)
        self.fc1 = Linear1D_Col.from_native_module(
            deepcopy(fc1), process_group=tp_group, gather_output=False, overlap=True
        )
        self.fc2 = Linear1D_Row.from_native_module(deepcopy(fc2), process_group=tp_group, parallel_input=True)

    def forward(self, x):
        return self.fc2(self.fc1(self.fc0(x)))


def force_assign_grad(p, g_dtype, grad=None):
    """avoid inconsistent grad and param dtype error"""
    orig_p = p.data
    p.data = torch.randn_like(p, device="cuda", dtype=g_dtype) if grad == None else grad
    p.grad = p.data
    p.data = orig_p


def set_dist_grad(
    dist_module: nn.Module,
    torch_model: nn.Module,
    g_dtype: torch.dtype,
    group: dist.ProcessGroup,
) -> None:
    """
    Set grads chunks for Tensor Parallel or ZeRO DP.
    We do not need a separate treatment for ZeRO,
    as the LowLevelOptimizer takes care of reduce-scattering grads.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    for p, torch_p in zip(dist_module.parameters(), torch_model.parameters()):
        if torch_p.grad is None:
            # avoid inconsistent grad and param dtype error
            force_assign_grad(torch_p, g_dtype)
        else:
            torch_p.grad += torch.randn_like(torch_p, device="cuda", dtype=g_dtype)

        if p.grad is None:
            force_assign_grad(p, g_dtype)

        if is_distributed_tensor(p):
            split_dim = get_split_dim(p)
            # Add grads only to the correctly split chunk
            force_assign_grad(p, g_dtype, torch_p.grad.chunk(world_size, dim=split_dim)[rank])
            # assert_close(p.grad, torch_p.grad.chunk(world_size, dim=split_dim)[rank])
        else:
            force_assign_grad(p, g_dtype, torch_p.grad)


@parameterize("p_g_dtype", _ALLOWED_P_G_TYPES)
@parameterize("bias_correction", [False, True])
@parameterize("tp_zero_size", [(1, 4), (4, 1), (2, 2)])
def run_dist_lamb_basic(
    bias_correction: bool, p_g_dtype: tuple[torch.dtype, torch.dtype], tp_zero_size: tuple[int, int]
) -> None:
    """Test without forward"""
    p_dtype, g_dtype = p_g_dtype
    tp_size, zero_size = tp_zero_size

    # Set distributed groups
    rank = dist.get_rank()
    clear_layout_converter()  # Ensure correct sharding
    proc_mesh = ProcessGroupMesh(tp_size, zero_size)
    tp_group = proc_mesh.get_group_along_axis(0)

    tp_rank = dist.get_rank(tp_group)
    torch.cuda.manual_seed(_SEED)  # Fix model init
    torch_model = Net(identity=True).to(rank)
    tp_model = TPNet(torch_model.fc0, torch_model.fc1, torch_model.fc2, tp_group).to(rank)
    # Ensure equal weight init
    assert_close(
        torch_model.fc1.weight[tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc1.weight,
    )
    assert_close(
        torch_model.fc2.weight[:, tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc2.weight,
    )

    # Set up optimizers
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    torch_optim = Lamb(
        setup_param_groups(torch_model), lr=lr, betas=(beta1, beta2), eps=eps, bias_correction=bias_correction
    )
    optim = DistributedLamb(
        setup_param_groups(tp_model),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        bias_correction=bias_correction,
    )
    optim.setup_distributed(tp_group)

    rtol, atol = 8e-5, 8e-5
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 2e-4, 2e-4
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 4e-4, 4e-4

    for i in range(_N_STEP):
        torch.cuda.manual_seed(_SEED)  # NOTE: having only one manual_seed above doesn't work?
        set_dist_grad(tp_model, torch_model, g_dtype, tp_group)

        torch_optim.step()
        optim.step()
        torch_optim.zero_grad()
        optim.zero_grad()
        try:
            assert_distributed_close(tp_model, torch_model, rtol, atol, tp_group)
        except Exception as e:
            _COORD.print_on_master(
                f"step {i + 1}: bias_correction: {bias_correction}, p_g_dtype: {p_g_dtype}, tp_zero_size: {tp_zero_size}"
            )
            raise e


@parameterize("p_g_dtype", _ALLOWED_P_G_TYPES)
@parameterize("bias_correction", [False, True])
@parameterize("tp_zero_size", [(2, 2), (4, 1), (1, 4)])
def run_dist_lamb_fwd_bwd(
    bias_correction: bool, p_g_dtype: tuple[torch.dtype, torch.dtype], tp_zero_size: tuple[int, int]
) -> None:
    p_dtype, g_dtype = p_g_dtype
    tp_size, zero_size = tp_zero_size

    # Set distributed groups
    rank = dist.get_rank()
    proc_mesh = ProcessGroupMesh(tp_size, zero_size)
    tp_group = proc_mesh.get_group_along_axis(0)
    dp_group = proc_mesh.get_group_along_axis(1)
    tp_rank = dist.get_rank(tp_group)

    torch.cuda.manual_seed(_SEED)
    clear_layout_converter()  # Ensure correct sharding
    torch_model = Net().to(rank)
    tp_model = TPNet(torch_model.fc0, torch_model.fc1, torch_model.fc2, tp_group).to(rank)

    assert_close(
        torch_model.fc1.weight[tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc1.weight,
    )
    assert_close(
        torch_model.fc2.weight[:, tp_rank * _HID_DIM // tp_size : (tp_rank + 1) * _HID_DIM // tp_size],
        tp_model.fc2.weight,
    )

    # Set up optimizers
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    torch_optim = Lamb(
        setup_param_groups(torch_model), lr=lr, betas=(beta1, beta2), eps=eps, bias_correction=bias_correction
    )
    optim = DistributedLamb(
        setup_param_groups(tp_model),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        bias_correction=bias_correction,
    )

    # Setup distributed optimizer
    if zero_size > 1:
        optim = LowLevelZeroOptimizer(
            optim,
            overlap_communication=True,
            initial_scale=128,
            partition_grad=True,
            dp_process_group=dp_group,
            verbose=True,
        )
        shard_to_param = optim._param_store.master_to_working_param
        torch_optim = LowLevelZeroOptimizer(
            torch_optim,
            overlap_communication=True,
            initial_scale=128,
            partition_grad=True,
            dp_process_group=dp_group,
            verbose=True,
        )
        optim.optim.setup_distributed(tp_group, dp_group, shard_to_param)
    else:
        optim.setup_distributed(tp_group)

    rtol, atol = 3e-5, 3e-5
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 1e-4, 1e-4
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 2e-4, 2e-4

    torch.cuda.manual_seed(_SEED)  # NOTE: having only one manual_seed above doesn't work?
    x = torch.randn(_BS, _IN_DIM, dtype=p_dtype, device=rank)

    out_tp = tp_model(x)
    out = torch_model(x)
    try:
        assert_close(out, out_tp, rtol=rtol, atol=atol)
    except Exception as e:
        _COORD.print_on_master(
            f"bias_correction: {bias_correction}, p_g_dtype: {p_g_dtype}, tp_zero_size: {tp_zero_size}"
        )
        raise e

    if zero_size > 1:
        optim.backward(out_tp.sum())
        torch_optim.backward(out.sum())
    else:
        out_tp.sum().backward()
        out.sum().backward()

    torch_optim.step()
    optim.step()
    torch_optim.zero_grad()
    optim.zero_grad()
    try:
        assert_distributed_close(tp_model, torch_model, rtol, atol, tp_group)
    except Exception as e:
        _COORD.print_on_master(
            f"bias_correction: {bias_correction}, p_g_dtype: {p_g_dtype}, tp_zero_size: {tp_zero_size}"
        )
        raise e


def check_dist_lamb(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    global _COORD
    _COORD = DistCoordinator()

    run_dist_lamb_basic()
    _COORD.print_on_master("Basic tests passed")

    run_dist_lamb_fwd_bwd()
    _COORD.print_on_master("Forward-backward tests passed")

    run_bert_test(optim_class=Lamb, sharded_optim_class=DistributedLamb)
    print(f"rank {rank} tests passed :)")


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_lamb():
    spawn(check_dist_lamb, nprocs=4)


if __name__ == "__main__":
    test_dist_lamb()
