from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from colossal_moe.models.mixtral_checkpoint import MixtralMoEHybridParallelCheckpointIO
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from torch.optim import Adam
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing.utils import spawn

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


def check_model_equal(model1, model2):
    assert set(model1.state_dict().keys()) == set(model2.state_dict().keys())
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1.half(), p2.half())


def get_optimizer_snapshot(optim):
    state = {id(k): deepcopy(v) for k, v in optim.state.items()}
    param_groups = []
    for group in optim.param_groups:
        params = [id(p) for p in group["params"]]
        new_group = {"params": params}
        for k, v in group.items():
            if k != "params":
                new_group[k] = v
        param_groups.append(new_group)
    return {
        "state": state,
        "param_groups": param_groups,
    }


def check_optimizer_snapshot_equal(snapshot1, snapshot2):
    # check param_groups
    assert len(snapshot1["param_groups"]) == len(snapshot2["param_groups"])
    for group1, group2 in zip(snapshot1["param_groups"], snapshot2["param_groups"]):
        assert set(group1.keys()) == set(group2.keys())
        for k in group1.keys():
            assert group1[k] == group2[k]
    # check state
    assert set(snapshot1["state"].keys()) == set(
        snapshot2["state"].keys()
    ), f"{snapshot1['state'].keys()}, {snapshot2['state'].keys()}"
    for pid in snapshot1["state"].keys():
        state1, state2 = snapshot1["state"][pid], snapshot2["state"][pid]
        assert set(state1.keys()) == set(state2.keys())
        for k in state1.keys():
            if isinstance(state1[k], torch.Tensor):
                assert torch.equal(state1[k], state2[k]), f"{k}, {state1[k]}, {state2[k]}"
            else:
                assert state1[k] == state2[k]


def check_mixtral_moe_layer():
    torch.cuda.set_device(dist.get_rank())
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (2, tokens)).cuda()
    orig_model = MixtralForCausalLM(config).cuda()
    model = deepcopy(orig_model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    plugin = MoeHybridParallelPlugin(
        tp_size=1,
        pp_size=2,
        ep_size=2,
        custom_policy=MixtralForCausalLMPolicy(),
        checkpoint_io=MixtralMoEHybridParallelCheckpointIO,
        microbatch_size=1,
        zero_stage=1,
    )
    booster = Booster(plugin=plugin)
    model, optimizer, *_ = booster.boost(model=model, optimizer=optimizer)
    # initialize grads
    data_iter = iter(
        [{"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}]
    )
    booster.execute_pipeline(
        data_iter,
        model,
        lambda outputs, inputs: outputs.loss,
        optimizer,
    )

    # check save model
    booster.save_model(model, "mixtral_model", shard=True)
    dist.barrier()
    if dist.get_rank() == 0:
        saved_model = MixtralForCausalLM.from_pretrained("mixtral_model").cuda()
        check_model_equal(orig_model, saved_model)
        saved_model.save_pretrained("mixtral_hf_model")
    dist.barrier()

    # check load model
    new_model = MixtralForCausalLM(config).cuda()
    new_optimizer = Adam(new_model.parameters(), lr=1e-3)
    new_model, new_optimizer, *_ = booster.boost(model=new_model, optimizer=new_optimizer)
    booster.load_model(new_model, "mixtral_hf_model")
    check_model_equal(model, new_model)

    # check save optimizer
    optimizer.step()
    for group in optimizer.param_groups:
        group["lr"] = 0.1
    snapshot = get_optimizer_snapshot(optimizer.unwrap())
    booster.save_optimizer(optimizer, "mixtral_optim", shard=True)
    dist.barrier()
    # reset optimizer state
    for state in optimizer.unwrap().state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                v.zero_()
    booster.load_optimizer(optimizer, "mixtral_optim")
    loaded_snapshot = get_optimizer_snapshot(optimizer.unwrap())
    check_optimizer_snapshot_equal(snapshot, loaded_snapshot)


def run_dist(rank: int, world_size: int, port: int):
    colossalai.launch({}, rank, world_size, "localhost", port)
    check_mixtral_moe_layer()


@pytest.mark.parametrize("world_size", [4])
def test_mixtral_moe_layer(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_mixtral_moe_layer(4)
