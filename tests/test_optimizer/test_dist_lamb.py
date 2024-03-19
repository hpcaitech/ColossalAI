import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.nn.optimizer import DistributedLamb, Lamb
from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
from colossalai.tensor.d_tensor.sharding_spec import DimSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.zero import LowLevelZeroOptimizer

# _ALLOWED_OPTIM_DEVICES = [
#     (DistributedLamb, torch.device("cuda:0")),
# ]

_ALLOWED_P_G_TYPES = [
    (torch.float, torch.float),  # pure fp32
    (torch.float, torch.half),  # fp16 amp
    (torch.float, torch.bfloat16),  # bfloat16 amp
]

N_STEPS = 3
# Identifiers for Tensor Parallel linear layers
_TP_SPEC = DimSpec([0])
_IN_DIM = 32
_HID_DIM = 128


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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_IN_DIM, _HID_DIM)
        self.fc2 = nn.Linear(_HID_DIM, _IN_DIM)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TPNet(nn.Module):
    def __init__(self, fc1, fc2, tp_group=None):
        super().__init__()
        self.fc1 = Linear1D_Col.from_native_module(fc1, process_group=tp_group, gather_output=False, overlap=True)
        self.fc2 = Linear1D_Row.from_native_module(fc2, process_group=tp_group, parallel_input=True)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def set_dist_grad(
    dist_module: nn.Module, torch_model: nn.Module, g_dtype: torch.dtype, group: dist.ProcessGroup
) -> None:
    """
    Set split grads for Tensor Parallel or ZeRO DP.
    We do not need separate treatment for ZeRO,
    as the wrapper takes care of reduce-scattering grads.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    for p, torch_p in zip(dist_module.parameters(), torch_model.parameters()):
        if torch_p.grad is None:
            torch_p.grad = torch.zeros_like(torch_p)

        is_tp = hasattr(p, "dist_layout")
        if is_tp:
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


@parameterize("p_g_dtype", _ALLOWED_P_G_TYPES)
@parameterize("bias_correction", [False, True])
@parameterize("zero_size", [2])
@parameterize("tp_size", [2])
def run_dist_lamb_optim(
    bias_correction: bool,
    p_g_dtype: tuple[torch.dtype, torch.dtype],
    zero_size: int,
    tp_size: int,
    # device_mesh: DeviceMesh
) -> None:
    p_dtype, g_dtype = p_g_dtype
    device_mesh, *_ = DistributedLamb.init_distributed(tp_size, zero_size)
    tp_group = device_mesh.get_process_group(axis=0)
    dp_group = device_mesh.get_process_group(axis=1)

    torch_model = Net()
    model = TPNet(torch_model.fc1, torch_model.fc2, tp_group)

    lr = 1e-3
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    torch_optim = Lamb(setup_param_groups(torch_model), lr=lr, betas=(beta1, beta2), eps=eps)
    optim = DistributedLamb(
        setup_param_groups(model),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        bias_correction=bias_correction,
        device_mesh=device_mesh,
    )
    if zero_size > 1:
        optim = LowLevelZeroOptimizer(
            optim, overlap_communication=True, initial_scale=128, partition_grad=True, dp_process_group=dp_group
        )

    rtol, atol = 1e-5, 1e-5
    if p_dtype is torch.float16 or g_dtype is torch.float16:
        rtol, atol = 2e-3, 2e-3
    if p_dtype is torch.bfloat16 or g_dtype is torch.bfloat16:
        rtol, atol = 4e-3, 4e-3

    for _ in range(N_STEPS):
        set_dist_grad(model, torch_model, g_dtype, optim.optim.tp_group)
        torch_optim.step()
        optim.step()
        torch_optim.zero_grad()
        optim.zero_grad()
        for p, torch_p in zip(model.parameters(), torch_model.parameters()):
            # if overflow, the weight won't be updated. so there will be no nan in p
            assert not torch.isnan(p).any()
            assert torch.allclose(p.float(), torch_p, rtol=rtol, atol=atol)


def check_dist_lamb(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_dist_lamb_optim()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_lamb():
    spawn(check_dist_lamb, nprocs=4)


if __name__ == "__main__":
    test_dist_lamb()
