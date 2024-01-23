import shutil
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from colossal_moe.models.mixtral_checkpoint import MixtralMoEHybridParallelCheckpointIO
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from torch.optim import Adam
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.moe import MOE_MANAGER
from colossalai.testing.utils import spawn

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


def check_mixtral_moe_layer():
    torch.cuda.set_device(dist.get_rank())
    MOE_MANAGER.setup(parallel="EP", mode="fixed", fixed_dp_size=1, fixed_ep_size=2, fixed_pp_size=2)
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
        num_attention_heads=2,
    )
    torch.manual_seed(0)

    orig_model = MixtralForCausalLM(config).cuda()
    model = deepcopy(orig_model)
    replace_moe_layer(model)
    orig_optimizer = Adam(orig_model.parameters(), lr=1e-3)
    optimizer = Adam(model.parameters(), lr=1e-3)
    plugin = MoeHybridParallelPlugin(
        tp_size=1,
        pp_size=2,
        custom_policy=MixtralForCausalLMPolicy(),
        checkpoint_io=MixtralMoEHybridParallelCheckpointIO,
        microbatch_size=1,
        precision="fp32",
    )
    booster = Booster(plugin=plugin)
    model, optimizer, *_ = booster.boost(model=model, optimizer=optimizer)
    # check save model
    booster.save_model(model, "mixtral_ckpt", shard=True)
    dist.barrier()
    if dist.get_rank() == 0:
        saved_model = MixtralForCausalLM.from_pretrained("mixtral_ckpt").cuda()
        assert set(orig_model.state_dict().keys()) == set(saved_model.state_dict().keys())
        for p1, p2 in zip(orig_model.parameters(), saved_model.parameters()):
            assert torch.equal(p1, p2)
        shutil.rmtree("mixtral_ckpt")
        saved_model.save_pretrained("mixtral_ckpt")
    dist.barrier()
    # check load model
    new_model = MixtralForCausalLM(config).cuda()
    replace_moe_layer(new_model)
    new_optimizer = Adam(new_model.parameters(), lr=1e-3)
    new_model, new_optimizer, *_ = booster.boost(model=new_model, optimizer=new_optimizer)
    booster.load_model(new_model, "mixtral_ckpt/")
    assert set(model.state_dict().keys()) == set(new_model.state_dict().keys())
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)


def run_dist(rank: int, world_size: int, port: int):
    colossalai.launch({}, rank, world_size, "localhost", port)
    check_mixtral_moe_layer()


@pytest.mark.parametrize("world_size", [4])
def test_mixtral_moe_layer(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_mixtral_moe_layer(4)
