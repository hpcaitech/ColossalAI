import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import sync_moe_model_param
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from tests.test_moe.moe_utils import MoeGradientHandler, sync_local_from_ep, sync_tp_from_ep


def run_test(rank: int, world_size: int, port: int, num_experts: int, batch_size: int, dim: int, seed: int):
    assert batch_size % world_size == 0

    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    local_model = SparseMLP(num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2)
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    ep_model = SparseMLP(num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2)
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="TP")
    tp_model = SparseMLP(num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2)
    ep_model = ep_model.to(get_current_device())
    tp_model = tp_model.to(get_current_device())
    local_model = local_model.to(get_current_device())

    # sync ep param
    sync_moe_model_param(ep_model)
    dist_dict = MOE_MANAGER.parallel_info_dict
    assert_equal_in_group(ep_model.experts.wi.data, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.data, dist_dict[world_size].dp_group)
    grad_handler = MoeGradientHandler(ep_model)
    # sync tp param
    sync_tp_from_ep(tp_model, ep_model)
    # sync local param
    sync_local_from_ep(local_model, ep_model)

    rank = dist.get_rank()
    torch.cuda.manual_seed(seed)
    tp_data = torch.randn(batch_size, dim, device=get_current_device())
    micro_batch_size = batch_size // world_size
    ep_data = tp_data.detach()[micro_batch_size * rank : micro_batch_size * (rank + 1)]

    out_local = local_model(tp_data)
    MOE_MANAGER.reset_loss()
    out_tp = tp_model(tp_data)
    MOE_MANAGER.reset_loss()
    out_ep = ep_model(ep_data)
    MOE_MANAGER.reset_loss()
    assert torch.allclose(out_ep, out_tp[micro_batch_size * rank : micro_batch_size * (rank + 1)])
    assert torch.allclose(out_ep, out_local[micro_batch_size * rank : micro_batch_size * (rank + 1)])

    out_local.mean().backward()
    out_tp.mean().backward()
    out_ep.mean().backward()
    grad_handler.handle_gradient()

    assert_equal_in_group(ep_model.experts.wi.grad, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.grad, dist_dict[world_size].dp_group)

    sync_local_from_ep(local_model, ep_model, assert_grad_flag=True)
    sync_tp_from_ep(tp_model, ep_model, assert_grad_flag=True)


@pytest.mark.dist
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("dim", [32])
@pytest.mark.parametrize("seed", [42])
@rerun_if_address_is_in_use()
def test_moe_ep_tp(num_experts: int, batch_size: int, dim: int, seed: int):
    spawn(run_test, 2, num_experts=num_experts, batch_size=batch_size, dim=dim, seed=seed)


if __name__ == "__main__":
    test_moe_ep_tp(num_experts=8, batch_size=8, dim=256, seed=42)
