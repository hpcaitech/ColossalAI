import os
import tempfile
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch.optim import SGD, Adam
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import parameterize, spawn
from colossalai.testing.random import seed_all
from colossalai.testing.utils import spawn
from tests.test_moe.moe_utils import check_model_equal

tokens, n_experts = 7, 4
hidden_size = 8
top_k = 2


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


def check_optimizer_snapshot_equal(snapshot1, snapshot2, param2name, moe_dp_group=None):
    assert len(snapshot1["param_groups"]) == len(snapshot2["param_groups"])
    for group1, group2 in zip(snapshot1["param_groups"], snapshot2["param_groups"]):
        assert set(group1.keys()) == set(group2.keys())
        for k in group1.keys():
            assert group1[k] == group2[k]
    # check state
    assert set(snapshot1["state"].keys()) == set(
        snapshot2["state"].keys()
    ), f"{snapshot1['state'].keys()}, {snapshot2['state'].keys()}"

    passed = True
    count = 0
    for pid in snapshot1["state"].keys():
        state1, state2 = snapshot1["state"][pid], snapshot2["state"][pid]
        assert set(state1.keys()) == set(state2.keys())
        bug = False
        for k in state1.keys():
            if isinstance(state1[k], torch.Tensor):
                if not torch.equal(state1[k], state2[k]):
                    bug = True
                    count += 1
            else:
                assert state1[k] == state2[k]
        if bug:
            passed = False

    if not passed:
        raise AssertionError(f"A total of {count} optim states are not equal")


@parameterize(
    "test_config",
    [
        [
            MixtralConfig(
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_local_experts=n_experts,
                num_experts_per_tok=top_k,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_hidden_layers=2,
            ),
            MixtralForCausalLM,
        ],
    ],
)
def check_moe_checkpoint(test_config):
    dtype, precision = torch.float16, "fp16"
    config, model_cls = test_config
    torch.cuda.set_device(dist.get_rank())

    context = tempfile.TemporaryDirectory() if dist.get_rank() == 0 else nullcontext()
    with context as f:
        if dist.get_rank() == 0:
            broadcast_objects = [f]  # any picklable object
        else:
            broadcast_objects = [None]
        dist.broadcast_object_list(broadcast_objects, src=0)

        input_ids = torch.randint(0, 100, (2, tokens)).cuda()
        orig_model = model_cls(config).cuda().to(dtype)

        seed_all(10086)
        model = deepcopy(orig_model)
        optimizer = SGD(model.parameters(), lr=1e-3)
        plugin = MoeHybridParallelPlugin(
            pp_size=2, ep_size=2, tp_size=1, microbatch_size=1, zero_stage=1, precision=precision
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
        tmpdirname = broadcast_objects[0]
        model_dir = os.path.join(tmpdirname, "mixtral_model")
        hf_model_dir = os.path.join(tmpdirname, "mixtral_hf_model")
        optim_dir = os.path.join(tmpdirname, "mixtral_optim")

        booster.save_model(model, model_dir, shard=True)
        dist.barrier()
        if dist.get_rank() == 0:
            saved_model = model_cls.from_pretrained(model_dir).cuda().to(dtype)
            check_model_equal(orig_model, saved_model, dtype=dtype)
            saved_model.save_pretrained(hf_model_dir)
        dist.barrier()
        # check load model
        new_model = model_cls(config).cuda().to(dtype)
        new_optimizer = Adam(new_model.parameters(), lr=1e-3)
        new_model, new_optimizer, *_ = booster.boost(model=new_model, optimizer=new_optimizer)
        booster.load_model(new_model, hf_model_dir)
        check_model_equal(model, new_model, dtype=dtype)

        # check save optimizer
        optimizer.step()
        for group in optimizer.param_groups:
            group["lr"] = 0.1
        snapshot = get_optimizer_snapshot(optimizer.unwrap())
        booster.save_optimizer(optimizer, optim_dir, shard=True)
        dist.barrier()

        # reset optimizer state
        for state in optimizer.unwrap().state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    v.zero_()
        booster.load_optimizer(optimizer, optim_dir)
        loaded_snapshot = get_optimizer_snapshot(optimizer.unwrap())
        check_optimizer_snapshot_equal(snapshot, loaded_snapshot, None, model)
        # Ensure rank 0 waits for all other ranks to finish
        dist.barrier()


def run_dist(rank: int, world_size: int, port: int):
    colossalai.launch(rank, world_size, "localhost", port)
    check_moe_checkpoint()


# Test EP + ZeRO + PP
@pytest.mark.parametrize("world_size", [4])
def test_mixtral_moe_layer(world_size: int):
    spawn(run_dist, world_size)


if __name__ == "__main__":
    test_mixtral_moe_layer(4)
