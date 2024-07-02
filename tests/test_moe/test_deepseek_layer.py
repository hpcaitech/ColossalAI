from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.modeling.deepseek import EPDeepseekMoE
from colossalai.shardformer.modeling.deepseek_moe_16b_base.configuration_deepseek import DeepseekConfig
from colossalai.shardformer.modeling.deepseek_moe_16b_base.modeling_deepseek import DeepseekMoE
from colossalai.testing.utils import spawn

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


def check_deepseek_moe_layer():
    torch.cuda.set_device(dist.get_rank())
    plugin = MoeHybridParallelPlugin(
        precision="bf16",
        tp_size=1,
        pp_size=1,
        ep_size=dist.get_world_size(),
    )
    config = DeepseekConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        n_routed_experts=n_experts,
        num_experts_per_tok=top_k,
    )
    torch.manual_seed(0)
    orig_model = DeepseekMoE(config).cuda()
    x = torch.rand(1, tokens, hidden_size, requires_grad=True).cuda()
    orig_output = orig_model(x)
    model = deepcopy(orig_model)
    model = EPDeepseekMoE.from_native_module(model, ep_group=plugin.ep_group)
    ep_output = model(x)
    assert_close(orig_output, ep_output)
    orig_loss = orig_output.mean()
    orig_loss.backward()
    ep_loss = ep_output.mean()
    ep_loss.backward()
    assert_close(orig_loss, ep_loss)
    name_to_p = {n: p for n, p in orig_model.named_parameters()}
    for n, ep_p in model.named_parameters():
        p = name_to_p[n]
        if ep_p.grad is not None:
            assert_close(p.grad, ep_p.grad)


def run_dist(rank: int, world_size: int, port: int):
    colossalai.launch(rank, world_size, "localhost", port)
    check_deepseek_moe_layer()


# @pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("world_size", [2])
def test_deepseek_moe_layer(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_deepseek_moe_layer(2)
