from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import transformers
from packaging.version import Version
from torch.testing import assert_close

import colossalai
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize
from colossalai.testing.utils import spawn

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


@parameterize("ep_size", [2, 4])
@parameterize("norm_topk_prob", [True, False])
def check_qwen3_moe_layer(ep_size: int, norm_topk_prob: bool):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    from colossalai.shardformer.modeling.qwen3_moe import EPQwen3MoeSparseMoeBlock

    torch.cuda.set_device(dist.get_rank())
    plugin = MoeHybridParallelPlugin(
        precision="fp32",
        tp_size=1,
        pp_size=1,
        zero_stage=1,
        ep_size=ep_size,
    )
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        moe_intermediate_size=hidden_size * 2,
        num_experts=n_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=norm_topk_prob,
    )
    torch.manual_seed(0)
    orig_model = Qwen3MoeSparseMoeBlock(config).cuda()
    # a larger init than the default initializer_range so that the expert outputs are not negligible
    for p in orig_model.parameters():
        torch.nn.init.normal_(p, std=0.5)
    # as in training, the input requires grad, otherwise an ep rank whose experts get no tokens
    # would skip the backward all-to-all
    x = torch.rand(1, tokens, hidden_size, device="cuda", requires_grad=True)
    ep_x = x.detach().clone().requires_grad_()
    orig_output, orig_logits = orig_model(x)
    model = deepcopy(orig_model)
    model = EPQwen3MoeSparseMoeBlock.from_native_module(
        model,
        ep_group=plugin.ep_group,
        tp_group=plugin.tp_group,
        moe_dp_group=plugin.moe_dp_group,
    )
    assert sum(e.gate_proj.weight is not None for e in model.experts) == n_experts // ep_size

    ep_output, ep_logits = model(ep_x)
    assert_close(orig_logits, ep_logits)
    assert_close(orig_output, ep_output)

    orig_output.pow(2).sum().backward()
    ep_output.pow(2).sum().backward()
    assert_close(x.grad, ep_x.grad)
    name_to_p = {n: p for n, p in orig_model.named_parameters()}
    for n, ep_p in model.named_parameters():
        if ep_p.grad is not None:
            assert_close(name_to_p[n].grad, ep_p.grad, msg=lambda m: f"{n}: {m}")


def run_dist(rank: int, world_size: int, port: int):
    colossalai.launch(rank, world_size, "localhost", port)
    check_qwen3_moe_layer()


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("4.51.0"), reason="Requires transformers version 4.51.0 or later"
)
@pytest.mark.parametrize("world_size", [4])
def test_qwen3_moe_layer(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_qwen3_moe_layer(4)
