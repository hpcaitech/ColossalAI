import importlib
import os
import shutil
import sys

import pytest
import torch
import torch.distributed as dist
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.testing import DummyDataloader, check_state_dict_equal, rerun_if_address_is_in_use, spawn

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "examples/language/openmoe",
    )
)

OpenMoeForCausalLM = importlib.import_module("model.modeling_openmoe").OpenMoeForCausalLM
set_openmoe_args = importlib.import_module("model.modeling_openmoe").set_openmoe_args
OpenMoeForCausalLMPolicy = importlib.import_module("model.openmoe_policy").OpenMoeForCausalLMPolicy


def data_gen_fn(batch_size: int = 2, max_length: int = 4, vocab_size: int = 20):
    input_ids = torch.randint(0, vocab_size, (batch_size, max_length), device=get_accelerator().get_current_device())
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }


def run_fwd_bwd(
    model, data, label, criterion, optimizer, enable_autocast=False, pipeline=False, booster=None, plugin=None
):
    model.train()
    if pipeline:
        train_dataloader_iter = DummyDataloader(data_gen_fn, length=1)
        is_pp_last_stage = booster.plugin.stage_manager.is_last_stage()
        y = booster.execute_pipeline(
            train_dataloader_iter,
            model,
            lambda x, y: x.loss,
            optimizer,
            return_loss=True,
        )
        # Backward and optimize
        if is_pp_last_stage:
            loss = y["loss"]
    else:
        if criterion:
            y = model(data).logits
            loss = criterion(y)
        else:
            loss = model(data, label)
        loss = loss.float()

        if optimizer is not None:
            optimizer.backward(loss)
        else:
            loss.backward()
    return y


def get_config():
    config = LlamaConfig(
        vocab_size=300,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        dropout_rate=0.0,
        hidden_act="swiglu",
    )
    set_openmoe_args(config, num_experts=8, moe_layer_interval=1)
    return config


def get_model(parallel):
    config = get_config()
    model = OpenMoeForCausalLM(config)
    optim = torch.optim.Adam(model.parameters())

    if parallel == None:
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=1,
            ep_size=1,
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "ep":
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=1,
            ep_size=dist.get_world_size(),
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "ep_zero":
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=1,
            ep_size=2,
            zero_stage=2,
            moe_dp_size=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "hybrid":
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=2,
            ep_size=2,
            zero_stage=1,
            microbatch_size=1,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    booster = Booster(plugin=plugin)
    model, optim, _, _, _ = booster.boost(model=model, optimizer=optim)
    return model, booster, optim


def _test_moe_checkpoint(rank, parallel):
    model1, booster1, optim1 = get_model(parallel)
    model2, booster2, optim2 = get_model(parallel)
    model3, booster3, optim3 = get_model(parallel)

    # param ckpt
    # shard
    booster1.save_model(model1, "./tmp_ckpt1", shard=True, size_per_shard=1)
    booster2.load_model(model2, "./tmp_ckpt1")
    # unshard
    booster1.save_model(model1, "./tmp_ckpt1.pth")
    booster3.load_model(model3, "./tmp_ckpt1.pth")
    # check
    check_state_dict_equal(model1.state_dict(), model2.state_dict(), False)
    check_state_dict_equal(model1.state_dict(), model3.state_dict(), False)

    # optim ckpt
    criterion = lambda x: x.mean()
    data = torch.randint(0, 4, (2, 4)).cuda()
    label = torch.randint(0, 4, (2,)).cuda()
    if parallel == "hybrid":
        kwargs = {"pipeline": True, "booster": booster1, "plugin": booster1.plugin}
    else:
        kwargs = {}
    run_fwd_bwd(model1, data, label, criterion, optim1, **kwargs)
    optim1.step()
    optim1.zero_grad()
    # shard
    booster1.save_optimizer(optim1, "./tmp_ckpt2", shard=True, size_per_shard=1)
    dist.barrier()
    booster2.load_optimizer(optim2, "./tmp_ckpt2")
    # unshard
    booster1.save_optimizer(optim1, "./tmp_ckpt2.pth")
    booster3.load_optimizer(optim3, "./tmp_ckpt2.pth")
    # check
    check_state_dict_equal(optim1.optim.state_dict(), optim2.optim.state_dict(), False)
    check_state_dict_equal(optim1.optim.state_dict(), optim3.optim.state_dict(), False)

    if dist.get_rank() == 0:
        shutil.rmtree("./tmp_ckpt1")
        shutil.rmtree("./tmp_ckpt2")
        os.remove("./tmp_ckpt1.pth")
        os.remove("./tmp_ckpt2.pth")


def _run_dist(rank, world_size, port, parallel):
    colossalai.launch(
        config=dict(),
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    _test_moe_checkpoint(rank, parallel)


# @pytest.mark.skip(reason="This is tested in ColossalMOE")
@pytest.mark.dist
@pytest.mark.parametrize("world_size", [4])
@pytest.mark.parametrize("parallel", [None, "ep", "ep_zero", "hybrid"])
@rerun_if_address_is_in_use()
def test_moe_checkpoint(world_size, parallel):
    spawn(_run_dist, world_size, parallel=parallel)


if __name__ == "__main__":
    test_moe_checkpoint(world_size=4, parallel="hybrid")
