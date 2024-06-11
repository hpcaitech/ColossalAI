from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.modeling.mixtral import EPMixtralSparseMoeBlock
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.testing.random import seed_all
from colossalai.zero import LowLevelZeroOptimizer
from tests.test_moe.moe_utils import loose_close

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


def split_grad(grad, world_size):
    with torch.no_grad():
        grad = grad.clone().detach().flatten()
        padding_size = (world_size - grad.numel() % world_size) % world_size
        if padding_size > 0:
            grad = torch.nn.functional.pad(grad, [0, padding_size])
        splited_grad = grad.split(grad.numel() // world_size)
    return splited_grad


@parameterize("dtype", [torch.float16, torch.bfloat16])
@parameterize("master_weights", [True, False])
@parameterize("stage", [1, 2])
def run_zero_with_original_model(world_size, master_weights: bool, dtype: torch.dtype, stage: int):
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(dist.get_rank())
    plugin = MoeHybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        ep_size=dist.get_world_size() // 2,
    )

    seed_all(10086)
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
    )

    orig_model = MixtralSparseMoeBlock(config).to(dtype).cuda()

    ori_model = DDP(orig_model.cuda(), static_graph=True).cuda()

    zero_model = deepcopy(orig_model)
    zero_model = EPMixtralSparseMoeBlock.from_native_module(zero_model, ep_group=plugin.ep_group)

    zero_optimizer = torch.optim.SGD(zero_model.parameters(), lr=1)
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        overlap_communication=True,
        initial_scale=1,
        reduce_bucket_size=1024 * 1024,
        master_weights=master_weights,
        moe_extra_dp_process_group=plugin.moe_dp_group,
        partition_grad=(stage == 2),
    )

    ori_optimizer = torch.optim.SGD(ori_model.parameters(), lr=1)

    # create
    seed_all(1453 + rank)
    input_data = torch.rand(1, tokens, hidden_size, requires_grad=True).cuda()
    # zero-dp forward
    zero_output, zero_logits = zero_model(input_data.to(dtype))

    # torch-ddp forward
    ori_output, ori_logits = ori_model(input_data.to(dtype))
    loose_close(zero_output, ori_output, dtype=dtype)

    # zero-dp backward
    zero_optimizer.backward(zero_output.mean().float())

    # torch-ddp backward
    ori_output.mean().float().backward()

    # check grad
    name_to_p = {n: p for n, p in ori_model.module.named_parameters()}

    for n, p in zero_model.named_parameters():
        if is_moe_tensor(p):  # moe param
            if p.grad is None:
                """
                For fixed input seed, the test input may cause a certain expert not to be routed to,
                so its gradient is None instead of a tensor, which may lead to a potential bug.
                TODO(haze188) fix later
                """
                p.grad = torch.zeros_like(p)
                continue
            dist.all_reduce(
                p.grad, group=plugin.moe_dp_group
            )  # TODO(haze188) bug fix: this step should be finished by zero
            p.grad = (
                p.grad / plugin.moe_dp_group.size()
            )  # moe param scaling amoung the moe dp group, not the WORLD group.
            loose_close(p.grad, name_to_p[n].grad, dtype=dtype)
            continue
        else:
            zero_grad_list = zero_optimizer._grad_store.get_partitioned_gradients_by_param_id(0, id(p))
            assert len(zero_grad_list) != 0
        ori_grad_list = split_grad(name_to_p[n].grad, world_size)
        if stage == 2:
            # Zero2 splits the gradient, and each rank holds the corresponding part
            ori_grad_list = ori_grad_list[rank : rank + 1]
        for zero_grad, torch_grad in zip(zero_grad_list, ori_grad_list):
            loose_close(zero_grad, torch_grad, dtype=dtype)

    # zero-dp step
    zero_optimizer.step()

    # original model step
    ori_optimizer.step()

    # check updated param
    for n, p in zero_model.named_parameters():
        loose_close(p.data, name_to_p[n].data, dtype=dtype)


def run_dist(rank, world_size, port):
    colossalai.launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_zero_with_original_model(world_size=world_size)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [2, 4])
@rerun_if_address_is_in_use()
def test_moe_zero_model(world_size):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_moe_zero_model(world_size=2)
