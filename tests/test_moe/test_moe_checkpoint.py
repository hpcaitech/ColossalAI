import importlib
import os
import shutil
import sys

import pytest
import torch
import torch.distributed as dist
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import rerun_if_address_is_in_use, spawn

sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "examples/language/openmoe",
))

OpenMoeForCausalLM = importlib.import_module("model.modeling_openmoe").OpenMoeForCausalLM
set_openmoe_args = importlib.import_module("model.modeling_openmoe").set_openmoe_args
OpenMoeForCausalLMPolicy = importlib.import_module("model.openmoe_policy").OpenMoeForCausalLMPolicy


def get_config():
    config = LlamaConfig(
        vocab_size=300,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        head_dim=4,
        dropout_rate=0.0,
        hidden_act="swiglu",
    )
    set_openmoe_args(config, num_experts=16, moe_layer_interval=1)
    return config


def get_model(parallel):
    config = get_config()
    model = OpenMoeForCausalLM(config)

    if parallel == None:
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=0,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "zero_ep":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "hybrid":
        plugin = MoeHybridParallelPlugin(
            tp_size=1,
            pp_size=2,
            zero_stage=1,
            microbatch_size=1,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    booster = Booster(plugin=plugin)
    model, _, _, _, _ = booster.boost(model=model)
    return model, booster


def _test_moe_checkpoint(parallel, shard):
    if parallel == None:
        MOE_MANAGER.setup(
            seed=42,
            parallel=None,
        )
    elif parallel == "zero2_ep":
        MOE_MANAGER.setup(
            seed=42,
            parallel="EP",
        )
    elif parallel == "hybrid":
        MOE_MANAGER.setup(
            seed=42,
            parallel="EP",
            mode="fixed",
            fixed_dp_size=1,
            fixed_ep_size=2,
            fixed_pp_size=2,
        )
    model1, booster1 = get_model(parallel)
    model2, booster2 = get_model(parallel)

    if shard:
        booster1.save_model(model1, "./tmp_ckpt", shard=True, size_per_shard=1)
        booster2.load_model(model2, "./tmp_ckpt")
    else:
        booster1.save_model(model1, "tmp_ckpt.pth")
        booster2.load_model(model2, "tmp_ckpt.pth")

    state1 = model1.state_dict()
    state2 = model2.state_dict()
    for k, v in state1.items():
        u = state2.get(k)
        assert torch.equal(u.data, v.data)

    if dist.get_rank() == 0:
        if shard:
            shutil.rmtree("./tmp_ckpt")
        else:
            os.remove("tmp_ckpt.pth")


def _run_dist(rank, world_size, port, parallel, shard):
    colossalai.launch(
        config=dict(),
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    _test_moe_checkpoint(parallel, shard)


@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("parallel", [None, "zero_ep", "hybrid"])
@pytest.mark.parametrize("shard", [True, False])
@rerun_if_address_is_in_use()
def test_moe_checkpoint(world_size, parallel, shard):
    spawn(_run_dist, world_size, parallel=parallel, shard=shard)


if __name__ == "__main__":
    test_moe_checkpoint(world_size=4, parallel="hybrid", shard=True)
