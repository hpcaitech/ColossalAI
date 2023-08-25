import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.nn.layer.moe import SparseMLP
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from colossalai.utils.moe import sync_moe_model_param
from tests.test_moe.moe_utils import MoeGradientHandler, sync_tp_from_ep

BATCH_SIZE = 4
DIM = 4


def run_test(rank, world_size, port):
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    MOE_CONTEXT.setup(42)    # MOE initialization

    ep_model = SparseMLP(num_experts=4, expert_parallel="EP", hidden_size=DIM, intermediate_size=DIM)
    tp_model = SparseMLP(num_experts=4, expert_parallel="TP", hidden_size=DIM, intermediate_size=DIM)
    ep_model = ep_model.to(get_current_device())
    tp_model = tp_model.to(get_current_device())

    # sync ep param
    sync_moe_model_param(ep_model)
    dist_dict = MOE_CONTEXT.parallel_info_dict
    assert_equal_in_group(ep_model.experts.wi.data, dist_dict[2].dp_group)
    assert_equal_in_group(ep_model.experts.wo.data, dist_dict[2].dp_group)
    grad_handler = MoeGradientHandler(ep_model)
    # sync tp param
    sync_tp_from_ep(tp_model, ep_model)

    rank = dist.get_rank()
    torch.cuda.manual_seed(78)
    tp_data = torch.randn(BATCH_SIZE, DIM, device=get_current_device())
    ep_data = tp_data.detach()[2 * rank:2 * (rank + 1)]

    out_tp = tp_model(tp_data)[0]
    MOE_CONTEXT.reset_loss()
    out_ep = ep_model(ep_data)[0]
    MOE_CONTEXT.reset_loss()
    assert torch.allclose(out_ep, out_tp[2 * rank:2 * (rank + 1)])

    out_tp.mean().backward()
    out_ep.mean().backward()
    grad_handler.handle_gradient()

    assert_equal_in_group(ep_model.experts.wi.grad, dist_dict[2].dp_group)
    assert_equal_in_group(ep_model.experts.wo.grad, dist_dict[2].dp_group)

    sync_tp_from_ep(tp_model, ep_model, assert_grad_flag=True)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_moe_ep_tp():
    spawn(run_test, 2)


if __name__ == '__main__':
    test_moe_ep_tp()
