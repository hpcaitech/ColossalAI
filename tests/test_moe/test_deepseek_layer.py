from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close
from transformers import AutoConfig, AutoModel

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.shardformer.modeling.deepseek import EPDeepseekMoE
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
        zero_stage=1,
        ep_size=dist.get_world_size(),
    )

    config = AutoConfig.from_pretrained(
        "deepseek-ai/deepseek-moe-16b-base",
        num_hidden_layers=1,
        n_routed_experts=n_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        first_k_dense_replace=0,
        num_attention_heads=2,
        trust_remote_code=True,
    )
    torch.manual_seed(0)
    # get the moe layer in auto model
    orig_model = AutoModel.from_config(config, trust_remote_code=True).layers[0].mlp.cuda()
    x = torch.rand(1, tokens, hidden_size, requires_grad=True).cuda()
    orig_output = orig_model(x)
    model = deepcopy(orig_model)
    model = EPDeepseekMoE.from_native_module(
        model,
        ep_group=plugin.ep_group,
        moe_dp_group=plugin.moe_dp_group,
        tp_group=plugin.tp_group,
    )
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


@pytest.mark.skip("tested in corresponding sharderformer")
@pytest.mark.parametrize("world_size", [2])
def test_deepseek_moe_layer(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_deepseek_moe_layer(2)
