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
from colossalai.zero.low_level.low_level_strategy import LowLevelOptStrategy, MoeZeroStrategy
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
    zero_params = list(filter(lambda x: not is_moe_tensor(x), zero_model.parameters()))
    moe_params = list(filter(lambda x: is_moe_tensor(x), zero_model.parameters()))
    zero_optimizer.param_groups.clear()
    zero_optimizer.add_param_group({"params": zero_params})
    zero_optimizer.add_param_group({"params": moe_params})
    strategies = [
        LowLevelOptStrategy(
            param_group=zero_optimizer.param_groups[0],
            dp_process_group=plugin.global_dp_group,
            overlap_communication=False,
            partition_grad=(stage == 2),
        ),
        MoeZeroStrategy(
            param_group=zero_optimizer.param_groups[1],
            dp_process_group=plugin.moe_dp_group,
            overlap_communication=True,
            partition_grad=(stage == 2),
        ),
    ]
    zero_optimizer = LowLevelZeroOptimizer(
        zero_optimizer,
        strategies,
        master_weights=master_weights,
        initial_scale=1,
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
        zero_grad = zero_optimizer.get_param_grad(p)
        if p.grad is None:
            """
            For fixed input seed, the test input may cause a certain expert not to be routed to,
            so its gradient is None instead of a tensor, which may lead to a potential bug.
            """
            # TODO(haze188) fix later
            p.grad = torch.zeros_like(p)
            continue
        loose_close(zero_grad, name_to_p[n].grad, dtype=dtype)

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
    test_moe_zero_model(world_size=4)
