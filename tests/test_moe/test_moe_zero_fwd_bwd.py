from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero import LowLevelZeroOptimizer
from tests.test_moe.moe_utils import MoeModel, loose_close


def split_ddp_grad(grad, world_size):
    with torch.no_grad():
        grad = grad.clone().detach().flatten()
        padding_size = (world_size - grad.numel() % world_size) % world_size
        if padding_size > 0:
            grad = torch.nn.functional.pad(grad, [0, padding_size])
        splited_grad = grad.split(grad.numel() // world_size)
    return splited_grad


tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


@parameterize("dtype", [torch.float16, torch.bfloat16])
@parameterize("master_weights", [True, False])
def run_zero_1_with_original_model(world_size, master_weights: bool, dtype: torch.dtype):
    torch.distributed.get_rank()

    torch.cuda.set_device(dist.get_rank())

    plugin = MoeHybridParallelPlugin(
        precision="bf16",
        tp_size=1,
        pp_size=1,
        ep_size=dist.get_world_size(),
    )

    seed_all(1453)
    zero_model = MoeModel(ep_group=plugin.ep_group).cuda().to(dtype)

    ori_model = deepcopy(zero_model).to(dtype)

    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        overlap_communication=True,
        initial_scale=1,
        reduce_bucket_size=1024 * 1024,
        master_weights=master_weights,
        moe_extra_dp_process_group=plugin.ep_group,
    )

    ori_optimizer = torch.optim.SGD(ori_model.parameters(), lr=1)

    # create
    input_data = torch.rand(1, 4).cuda()

    # zero-dp forward
    zero_output = zero_model(input_data.to(dtype))

    # torch-ddp forward
    ori_output = ori_model(input_data.to(dtype))
    loose_close(zero_output, ori_output, dtype=dtype)

    # zero-dp backward
    zero_optimizer.backward(zero_output.mean().float())

    # torch-ddp backward
    ori_output.mean().float().backward()

    # check grad
    for (n1, p1), (n2, p2) in zip(ori_model.named_parameters(), zero_model.named_parameters()):
        if dist.get_rank() == 0:
            print(n1, p1.shape, p1.grad is None, "\t", n2, p2.shape, p2.grad is None)

        if p1.grad is not None:
            if p2.grad is None:
                zero_grad_list = zero_optimizer._grad_store.get_partitioned_gradients_by_param_id(1, id(p2))
            else:  # moe param
                loose_close(p1.grad, p2.grad, dtype=dtype)
                continue

            ori_grad_list = split_ddp_grad(
                p1.grad, world_size
            )  # just flatten the original model grad to match the zero model grad shape
            for zero_grad, torch_grad in zip(zero_grad_list, ori_grad_list):
                loose_close(zero_grad, torch_grad, dtype=dtype)

    # zero-dp step
    zero_optimizer.step()

    # original model step
    ori_optimizer.step()

    # check updated param
    for (n, p), z1p in zip(ori_model.named_parameters(), zero_model.parameters()):
        loose_close(p.data, z1p.data, dtype=dtype)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_1_with_original_model(world_size=world_size)
    # run_zero_1_2()


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_model(world_size=2)
