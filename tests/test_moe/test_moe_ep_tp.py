import os
import warnings
from typing import Dict

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import sync_moe_model_param
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from tests.test_moe.moe_utils import MoeGradientHandler


def sync_tp_from_local(tp_model: SparseMLP, local_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from local model

    Args:
        tp_model (MoeModule)
        local_model (MoeModule)
    """
    for (tp_name, tp_param), (local_name, local_param) in zip(
        tp_model.named_parameters(), local_model.named_parameters()
    ):
        assert tp_name == local_name
        if not is_moe_tensor(tp_param):
            if assert_grad_flag:
                assert torch.allclose(tp_param, local_param)
                assert torch.allclose(tp_param.grad, local_param.grad)
            else:
                tp_param.data.copy_(local_param.data)
            continue

        tp_rank = get_ep_rank(tp_param)
        tp_dim = [i for i, (d1, d2) in enumerate(zip(tp_param.shape, local_param.shape)) if d1 != d2][0]
        tp_slice = [slice(None)] * tp_dim + [
            slice(tp_param.shape[tp_dim] * tp_rank, tp_param.shape[tp_dim] * (tp_rank + 1))
        ]

        if assert_grad_flag:
            assert torch.allclose(tp_param, local_param[tuple(tp_slice)])
            assert torch.allclose(tp_param.grad, local_param.grad[tuple(tp_slice)])
        else:
            tp_param.data.copy_(local_param[tuple(tp_slice)].data)


def sync_tp_from_ep(tp_model: SparseMLP, ep_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        tp_model (MoeModule)
        ep_model (MoeModule)
    """
    for (tp_name, tp_param), (ep_name, ep_param) in zip(tp_model.named_parameters(), ep_model.named_parameters()):
        assert tp_name == ep_name
        if not is_moe_tensor(tp_param):
            if assert_grad_flag:
                assert torch.allclose(tp_param, ep_param)
                assert torch.allclose(tp_param.grad, ep_param.grad)
            else:
                tp_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        # get tp param
        tp_dim = [i for i, (d1, d2) in enumerate(zip(tp_param.shape[1:], all_param.shape[1:])) if d1 != d2][0] + 1
        tp_rank = get_ep_rank(tp_param)
        tp_slice = [slice(None)] * tp_dim + [
            slice(tp_param.shape[tp_dim] * tp_rank, tp_param.shape[tp_dim] * (tp_rank + 1))
        ]
        new_tp_param = all_param[tuple(tp_slice)]
        if assert_grad_flag:
            new_grad = all_grad[tuple(tp_slice)]
        if assert_grad_flag:
            assert torch.allclose(tp_param, new_tp_param)
            assert torch.allclose(tp_param.grad, new_grad)
        else:
            tp_param.data.copy_(new_tp_param.data)


def sync_local_from_ep(local_model: SparseMLP, ep_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        local_model (MoeModule)
        ep_model (MoeModule)
    """
    for (local_name, local_param), (ep_name, ep_param) in zip(
        local_model.named_parameters(), ep_model.named_parameters()
    ):
        assert local_name == ep_name
        if "experts" not in local_name:
            if assert_grad_flag:
                assert torch.allclose(local_param, ep_param)
                assert torch.allclose(local_param.grad, ep_param.grad)
            else:
                local_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        if assert_grad_flag:
            assert torch.allclose(local_param, all_param)
            assert torch.allclose(local_param.grad, all_grad)
        else:
            local_param.data.copy_(all_param.data)


def run_test(rank: int, world_size: int, port: int, num_experts: int, batch_size: int, dim: int, config: Dict):
    assert batch_size % world_size == 0

    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    local_model = SparseMLP(num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2)
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    enable_hierarchical_comm = config.get("enable_hierarchical_comm", False)
    if enable_hierarchical_comm:
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    ep_model = SparseMLP(
        num_experts=num_experts,
        hidden_size=dim,
        intermediate_size=dim * 2,
        enable_hierarchical_comm=enable_hierarchical_comm,
    )
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="TP")
    tp_model = SparseMLP(num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2)
    ep_model = ep_model.to(get_accelerator().get_current_device())
    tp_model = tp_model.to(get_accelerator().get_current_device())
    local_model = local_model.to(get_accelerator().get_current_device())

    # sync ep param
    sync_moe_model_param(ep_model)
    dist_dict = MOE_MANAGER.parallel_info_dict
    assert_equal_in_group(ep_model.experts.wi.data, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.data, dist_dict[world_size].dp_group)
    ep_grad_handler = MoeGradientHandler(ep_model)
    # sync local param
    sync_local_from_ep(local_model, ep_model)
    # sync tp param
    sync_tp_from_ep(tp_model, ep_model)
    tp_grad_handler = MoeGradientHandler(tp_model)

    rank = dist.get_rank()
    input_data = torch.randn(batch_size, dim, device=get_accelerator().get_current_device())
    micro_batch_size = batch_size // world_size
    index = rank * micro_batch_size
    # NOTE: ep & tp takes in sharded data for each process
    shard_data = input_data.detach()[index : index + micro_batch_size]

    out_local = local_model(input_data)
    MOE_MANAGER.reset_loss()
    out_tp = tp_model(shard_data)
    MOE_MANAGER.reset_loss()
    out_ep = ep_model(shard_data)
    MOE_MANAGER.reset_loss()

    assert torch.allclose(
        out_tp, out_ep, atol=1e-6
    ), f"Rank {rank} failed, max diff: {torch.max(torch.abs(out_tp - out_ep))}"
    try:
        out_local_slice = out_local[index : index + micro_batch_size]
        assert torch.allclose(
            out_ep, out_local_slice, atol=1e-6
        ), f"Rank {rank} failed, max diff: {torch.max(torch.abs(out_ep - out_local_slice))}"
    except AssertionError:
        """
        e.g., in local model, tokens = 4, capacity = 2, experts = 2, topk = 1
            router yields [01] --> [0], [23] --> [1], this is valid as capacity is 2
            However, in ep mode, there are 2 separate routers dealing with sharded data.
            Assume router 0 handles token [01] and router 1 handles token [23].
            Note that for each router the capacity is only 1 !!!
            Thus, router 0 may yields [0] --> [0] or [1] --> [0], but not both.
            The same thing happens on router 1. And finally some tokens are dropped due to the sharded nature.
        """
        warnings.warn(
            "EP & TP may result in different behavior from local model. " "Please check the comments for details."
        )

    out_local.mean().backward()
    out_tp.mean().backward()
    tp_grad_handler.handle_gradient()
    out_ep.mean().backward()
    ep_grad_handler.handle_gradient()

    assert_equal_in_group(ep_model.experts.wi.grad, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.grad, dist_dict[world_size].dp_group)
    sync_tp_from_ep(tp_model, ep_model, assert_grad_flag=True)
    try:
        sync_local_from_ep(local_model, ep_model, assert_grad_flag=True)
    except AssertionError:
        warnings.warn(
            "EP & TP may result in different behavior from local model. " "Please check the comments for details."
        )


@pytest.mark.dist
@pytest.mark.parametrize("num_experts", [4, 64])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("dim", [64])
@pytest.mark.parametrize(
    "config",
    [
        {"enable_hierarchical_comm": False},
        {"enable_hierarchical_comm": True},
    ],
)
@rerun_if_address_is_in_use()
def test_moe_ep_tp(num_experts: int, batch_size: int, dim: int, config: Dict):
    spawn(run_test, 2, num_experts=num_experts, batch_size=batch_size, dim=dim, config=config)


if __name__ == "__main__":
    test_moe_ep_tp(num_experts=8, batch_size=32, dim=32)
