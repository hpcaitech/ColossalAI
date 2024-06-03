import pytest
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.legacy.tensor import ColoTensorSpec, ProcessGroup, distspec
from colossalai.legacy.utils.common import clip_grad_norm
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.colo_parameter import ColoParameter
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn


def close(num: float, other: float, rtol: float = 1e-5, atol: float = 1e-8):
    return abs(num - other) <= atol + rtol * other


def shard_param(p: ColoParameter) -> None:
    pg = p.get_process_group()
    p._redistribute(distspec.ShardSpec([0], [pg.tp_world_size()]))
    p.grad = p.grad.chunk(pg.tp_world_size(), 0)[pg.tp_local_rank()].clone().detach()


def check_grad_equal(p: Parameter, colo_p: ColoParameter) -> None:
    pg = colo_p.get_process_group()
    if p.shape != colo_p.shape:
        grad = p.grad.chunk(pg.tp_world_size(), 0)[pg.tp_local_rank()]
    else:
        grad = p.grad
    assert torch.allclose(grad, colo_p.grad), f"diff: {torch.abs(grad - colo_p.grad)}"


@parameterize("dtype", [torch.float])
@parameterize("device", ["mixed", "cuda", "cpu"])
@parameterize("norm_type", [2.0, 3.0, float("inf")])
def run_grad_clip_norm(world_size: int, dtype: torch.dtype, device: str, norm_type: float):
    print(f"{world_size}, {dtype}, {device}, {norm_type}")
    cuda_device = get_accelerator().get_current_device()
    devices = [cuda_device] * 4
    if device == "cpu":
        devices = [torch.device("cpu")] * 4
    elif device == "mixed":
        devices = [cuda_device] * 2 + [torch.device("cpu")] * 2
    pg = ProcessGroup(tp_degree=world_size)
    params = [Parameter(torch.empty(4, 4, dtype=dtype, device=devices[i])) for i in range(4)]
    colo_params = [
        ColoParameter(torch.empty(4, 4, dtype=dtype, device=devices[i]), spec=ColoTensorSpec(pg)) for i in range(4)
    ]
    for p, colo_p in zip(params, colo_params):
        grad = torch.rand_like(p)
        p.grad = grad
        colo_p.grad = grad.clone().detach()
    shard_param(colo_params[0])
    shard_param(colo_params[2])
    torch_norm = clip_grad_norm_(params, 1.0, norm_type=norm_type)
    colo_norm = clip_grad_norm(colo_params, 1.0, norm_type=norm_type)
    assert close(torch_norm, colo_norm), f"diff: {abs(torch_norm-colo_norm)}"
    for p, colo_p in zip(params, colo_params):
        check_grad_equal(p, colo_p)


def run_dist(rank, world_size, port):
    disable_existing_loggers()
    colossalai.legacy.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_grad_clip_norm(world_size=world_size)


@pytest.mark.skip("this need to be updated")
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_zero_clip_grad(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_zero_clip_grad(2)
